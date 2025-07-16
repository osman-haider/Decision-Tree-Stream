import numpy as np
from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu
from collections import Counter
import logging

# Configure logger for tracking progress and debugging
logger = logging.getLogger("DecisionStream")
logger.setLevel(logging.INFO)  # Change to DEBUG for more verbosity

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)

class DecisionStreamNode:
    """
    A node in the Decision Stream graph. Holds data indices, label values, references to parent and child nodes,
    split information, and prediction value.
    """
    def __init__(self, X, y, depth=0, parent=None):
        """
        Parameters:
        - X: np.ndarray, feature matrix of samples in this node.
        - y: np.ndarray, labels for samples in this node.
        - depth: int, depth level in the graph.
        - parent: DecisionStreamNode, parent node (None for root).
        """
        self.X = X
        self.y = y
        self.depth = depth
        self.parent = parent
        self.children = []
        self.is_terminal = False  # If True, node can't be split further
        self.split_feature = None  # Feature index used for split
        self.split_value = None    # Value at which split occurred
        self.prediction = self._majority_vote() if self._is_classification() else np.mean(y)

    def _is_classification(self):
        """
        Heuristically determines if the node's task is classification
        (few unique labels, integer labels).
        """
        return len(set(self.y)) < 20 and all(isinstance(yi, (int, np.integer)) for yi in self.y)

    def _majority_vote(self):
        """
        Computes the most common label (classification prediction).
        """
        return Counter(self.y).most_common(1)[0][0]

class DecisionStream:
    """
    Implements the Decision Stream learning algorithm:
    An iterative tree-like structure that splits and then merges statistically similar nodes,
    forming a deep DAG for classification or regression.
    """
    def __init__(self, max_depth=30, min_samples_split=10, plim=0.01, task='classification'):
        """
        Parameters:
        - max_depth: int, maximum depth to grow the stream.
        - min_samples_split: int, minimum samples required to split a node.
        - plim: float, significance threshold for merging nodes (higher means more merging).
        - task: str, 'classification' or 'regression'.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.plim = plim
        self.task = task
        self.root = None

    def fit(self, X, y):
        """
        Trains the Decision Stream on data X and labels y.
        1. Builds the stream iteratively by splitting and merging nodes.
        2. Stops when all leaves are terminal.
        """
        self.root = DecisionStreamNode(X, y, depth=0)
        leaves = [self.root]
        iteration = 0
        logger.info(
            f"Starting Decision Stream training: task={self.task}, plim={self.plim}, max_depth={self.max_depth}")
        while True:
            iteration += 1
            logger.info(f"Training iteration {iteration} - Number of leaves: {len(leaves)}")
            new_leaves = []
            split_count = 0
            terminal_count = 0

            # ---- Splitting phase ----
            for node in leaves:
                # Terminal if not enough samples, reached max depth, or previously marked
                if node.is_terminal or node.depth >= self.max_depth or len(node.y) < self.min_samples_split:
                    node.is_terminal = True
                    terminal_count += 1
                    logger.debug(f"Node at depth {node.depth} becomes terminal (samples={len(node.y)})")
                    new_leaves.append(node)
                else:
                    # Try to split node using statistical criteria
                    children = self._split_node(node)
                    if children:
                        split_count += 1
                        logger.info(
                            f"Splitting node at depth {node.depth} on feature {node.split_feature} at value {node.split_value}, creating {len(children)} children")
                        node.children = children
                        new_leaves.extend(children)
                    else:
                        node.is_terminal = True
                        terminal_count += 1
                        logger.debug(
                            f"Node at depth {node.depth} becomes terminal after split check (samples={len(node.y)})")
                        new_leaves.append(node)

            logger.info(f"Splits this iteration: {split_count}, Terminals: {terminal_count}")

            # ---- Merging phase ----
            merged_leaves = self._merge_leaves(new_leaves)
            if len(merged_leaves) != len(new_leaves):
                logger.info(f"Merge operation reduced leaves from {len(new_leaves)} to {len(merged_leaves)}")
            else:
                logger.debug(f"No merges performed this iteration.")

            # ---- End condition: all leaves terminal ----
            if all(n.is_terminal for n in merged_leaves):
                logger.info("All leaves are terminal. Training complete.")
                break
            leaves = merged_leaves

    def predict(self, X):
        """
        Makes predictions for a set of samples X.
        Returns:
        - np.ndarray, predicted values (class or regression output).
        """
        logger.info(f"Predicting on {X.shape[0]} samples.")
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        """
        Recursively traverses the stream for one sample to reach a leaf node
        and return its prediction.
        """
        if node.is_terminal or not node.children:
            return node.prediction
        for child in node.children:
            if self._follow_child(x, child, node):
                return self._predict_single(x, child)
        # If no child matched, return current node's prediction
        return node.prediction

    def _follow_child(self, x, child, parent):
        """
        Decides whether sample x should follow to 'child' from 'parent'
        node based on the parent's split rule.
        """
        if parent.split_feature is None:
            return True
        if isinstance(parent.split_value, (float, int)):
            # Binary split: left/right child
            return x[parent.split_feature] <= parent.split_value if child == parent.children[0] else x[
                                                                                                     parent.split_feature] > parent.split_value
        # For categorical splits (not implemented here, but logic placeholder)
        return x[parent.split_feature] in parent.split_value

    def _split_node(self, node):
        """
        Attempts to split a node into two children based on statistical difference of label distributions.
        Returns:
        - List of child DecisionStreamNodes, or None if no valid split found.
        """
        X, y = node.X, node.y
        best_score = np.inf
        best_feature = None
        best_value = None
        best_splits = None

        for f in range(X.shape[1]):
            unique_values = np.unique(X[:, f])
            for v in unique_values[:-1]:
                left_mask = X[:, f] <= v
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                y_left, y_right = y[left_mask], y[right_mask]
                p_value = self._stat_test(y_left, y_right)
                if p_value < self.plim and p_value < best_score:
                    best_score = p_value
                    best_feature = f
                    best_value = v
                    best_splits = [
                        (X[left_mask], y[left_mask]),
                        (X[right_mask], y[right_mask])
                    ]
        if best_splits is not None:
            logger.debug(
                f"Node at depth {node.depth} best split: feature {best_feature}, value {best_value}, p-value {best_score:.4g}")
            children = [DecisionStreamNode(xc, yc, depth=node.depth + 1, parent=node) for xc, yc in best_splits]
            node.split_feature = best_feature
            node.split_value = best_value
            return children
        logger.debug(f"Node at depth {node.depth}: no valid splits found.")
        return None

    def _merge_leaves(self, leaves):
        """
        Iteratively merges leaves that are statistically indistinguishable
        (p-value > plim) according to the label distributions.
        Returns a new list of merged leaves.
        """
        merged = []
        used = set()
        leaves = sorted(leaves, key=lambda n: len(n.y))
        i = 0
        while i < len(leaves):
            n1 = leaves[i]
            if i in used:
                i += 1
                continue
            found_merge = False
            for j in range(i + 1, len(leaves)):
                if j in used:
                    continue
                n2 = leaves[j]
                p_value = self._stat_test(n1.y, n2.y)
                if p_value > self.plim:
                    merged_X = np.vstack([n1.X, n2.X])
                    merged_y = np.concatenate([n1.y, n2.y])
                    new_node = DecisionStreamNode(merged_X, merged_y, depth=min(n1.depth, n2.depth))
                    merged.append(new_node)
                    used.add(j)
                    found_merge = True
                    logger.info(
                        f"Merged two nodes at depth {n1.depth} and {n2.depth} (samples: {len(n1.y)}+{len(n2.y)}) with p-value={p_value:.4g}")
                    break
            if not found_merge:
                merged.append(n1)
            i += 1
        return merged

    def _stat_test(self, y1, y2):
        """
        Applies the appropriate statistical test to determine if two sets of labels
        (from two nodes) are significantly different. Returns the p-value.
        """
        if self.task == 'classification':
            # Multiclass: use KS test; binary: Mann-Whitney U
            if len(np.unique(y1)) > 2 or len(np.unique(y2)) > 2:
                try:
                    return ks_2samp(y1, y2).pvalue
                except Exception as e:
                    logger.warning(f"KS test failed: {e}")
                    return 1.0
            else:
                try:
                    return mannwhitneyu(y1, y2, alternative='two-sided').pvalue
                except Exception as e:
                    logger.warning(f"Mann-Whitney U test failed: {e}")
                    return 1.0
        else:  # regression
            if len(y1) > 30 and len(y2) > 30:
                # Use Z-test for large samples
                mean1, mean2 = np.mean(y1), np.mean(y2)
                std1, std2 = np.std(y1), np.std(y2)
                n1, n2 = len(y1), len(y2)
                std_err = np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
                if std_err > 0:
                    from scipy.stats import norm
                    z = (mean1 - mean2) / std_err
                    return 2 * (1 - norm.cdf(abs(z)))
                else:
                    return 1.0
            else:
                # Use t-test for small samples
                try:
                    return ttest_ind(y1, y2, equal_var=False).pvalue
                except Exception as e:
                    logger.warning(f"T-test failed: {e}")
                    return 1.0
