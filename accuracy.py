from toy_example import *
from features import add_features
from sklearn.model_selection import KFold


def evaluate_pass_accuracy(model, X_LS, y_LS, features_list, n_folds=5):
    kf = KFold(
        n_splits=n_folds, shuffle=True, random_state=42
    )  # cross-validation setup
    accuracies = []

    for fold_num, (train_idx, val_idx) in enumerate(
        kf.split(X_LS), 1
    ):  # loop through each fold
        # split data
        # iloc selects rows from X_LS using train_idx
        X_train_fold = X_LS.iloc[train_idx]
        y_train_fold = y_LS.iloc[train_idx]
        X_val_fold = X_LS.iloc[val_idx]
        y_val_fold = y_LS.iloc[val_idx]

        # Process training fold
        print(f"  Creating {len(train_idx) * 22} training pairs...")
        X_train_pairs, y_train_pairs = make_pair_of_players(X_train_fold, y_train_fold)

        print(f"  Computing features...")
        X_train_pairs = add_features(X_train_pairs, X_original=X_train_fold)
        X_train_features = X_train_pairs[features_list]

        # Train
        print(f"  Training model on {len(X_train_features)} samples...")
        model.fit(X_train_features, y_train_pairs.values.ravel())
        print(f"  ✓ Training complete")

        # Process validation fold
        print(f"  Creating {len(val_idx) * 22} validation pairs...")
        X_val_pairs, _ = make_pair_of_players(X_val_fold)
        X_val_pairs = add_features(X_val_pairs, X_original=X_val_fold)
        X_val_features = X_val_pairs[features_list]

        # Predict
        print(f"  Predicting...")
        y_pred = model.predict_proba(X_val_features)[:, 1]
        probas = y_pred.reshape(X_val_fold.shape[0], 22)
        predictions = np.argmax(probas, axis=1) + 1

        # Evaluate
        accuracy = np.mean(predictions == y_val_fold.values.ravel())
        accuracies.append(accuracy)

        print(f"  ✓ Fold {fold_num} accuracy: {accuracy:.4f}")

    return np.array(accuracies)
