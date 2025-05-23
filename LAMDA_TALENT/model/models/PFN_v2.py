
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import Self, overload

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.preprocessing import LabelEncoder

from model.lib.pfn_v2.tabpfn.base import (
    create_inference_engine,
    determine_precision,
    initialize_tabpfn_model,
)
from model.lib.pfn_v2.tabpfn.constants import (
    PROBABILITY_EPSILON_ROUND_ZERO,
    SKLEARN_16_DECIMAL_PRECISION,
    ModelInterfaceConfig,
    XType,
    YType,
)
from model.lib.pfn_v2.tabpfn.preprocessing import (
    ClassifierEnsembleConfig,
    EnsembleConfig,
    default_classifier_preprocessor_configs,
)
from model.lib.pfn_v2.tabpfn.utils import (
    _fix_dtypes,
    _get_ordinal_encoder,
    infer_categorical_features,
    infer_device_and_type,
    infer_random_state,
    update_encoder_outlier_params,
    validate_X_predict,
    validate_Xy_fit,
)

from functools import partial
import numpy as np
import torch
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    check_is_fitted,
)

from model.lib.pfn_v2.tabpfn.base import (
    create_inference_engine,
    determine_precision,
    initialize_tabpfn_model,
)
from model.lib.pfn_v2.tabpfn.constants import (
    ModelInterfaceConfig,
    XType,
    YType,
)
from model.lib.pfn_v2.tabpfn.model.bar_distribution import FullSupportBarDistribution
from model.lib.pfn_v2.tabpfn.model.preprocessing import (
    ReshapeFeatureDistributionsStep,
)
from model.lib.pfn_v2.tabpfn.preprocessing import (
    EnsembleConfig,
    RegressorEnsembleConfig,
    default_regressor_preprocessor_configs,
)
from model.lib.pfn_v2.tabpfn.utils import (
    _fix_dtypes,
    _get_ordinal_encoder,
    _transform_borders_one,
    infer_categorical_features,
    infer_device_and_type,
    infer_random_state,
    translate_probs_across_borders,
    update_encoder_outlier_params,
    validate_X_predict,
    validate_Xy_fit,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from torch.types import _dtype

    from model.lib.pfn_v2.tabpfn.inference import (
        InferenceEngine,
    )
    from model.lib.pfn_v2.tabpfn.model.config import InferenceConfig

    try:
        from sklearn.base import Tags
    except ImportError:
        Tags = Any


class TabPFNClassifier(ClassifierMixin, BaseEstimator):
    """TabPFNClassifier class."""

    config_: InferenceConfig
    """The configuration of the loaded model to be used for inference."""

    interface_config_: ModelInterfaceConfig
    """Additional configuration of the interface for expert users."""

    device_: torch.device
    """The device determined to be used."""

    feature_names_in_: npt.NDArray[Any]
    """The feature names of the input data.

    May not be set if the input data does not have feature names,
    such as with a numpy array.
    """

    n_features_in_: int
    """The number of features in the input data used during `fit()`."""

    inferred_categorical_indices_: list[int]
    """The indices of the columns that were inferred to be categorical,
    as a product of any features deemed categorical by the user and what would
    work best for the model.
    """

    classes_: npt.NDArray[Any]
    """The unique classes found in the target data during `fit()`."""

    n_classes_: int
    """The number of classes found in the target data during `fit()`."""

    class_counts_: npt.NDArray[Any]
    """The number of classes per class found in the target data during `fit()`."""

    n_outputs_: Literal[1]
    """The number of outputs the model has. Only 1 for now"""

    use_autocast_: bool
    """Whether torch's autocast should be used."""

    forced_inference_dtype_: _dtype | None
    """The forced inference dtype for the model based on `inference_precision`."""

    executor_: InferenceEngine
    """The inference engine used to make predictions."""

    label_encoder_: LabelEncoder
    """The label encoder used to encode the target variable."""

    preprocessor_: ColumnTransformer
    """The column transformer used to preprocess the input data to be numeric."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_estimators: int = 4,
        categorical_features_indices: Sequence[int] | None = None,
        softmax_temperature: float = 0.9,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        model_path: str | Path | Literal["auto"] = "auto",
        device: str | torch.device | Literal["auto"] = "auto",
        ignore_pretraining_limits: bool = False,
        inference_precision: _dtype | Literal["autocast", "auto"] = "auto",
        fit_mode: Literal[
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
        ] = "fit_preprocessors",
        memory_saving_mode: bool | Literal["auto"] | float | int = "auto",
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
        n_jobs: int = -1,
        inference_config: dict | ModelInterfaceConfig | None = None,
    ) -> None:
        """A TabPFN interface for classification.

        Args:
            n_estimators:
                The number of estimators in the TabPFN ensemble. We aggregate the
                 predictions of `n_estimators`-many forward passes of TabPFN. Each
                 forward pass has (slightly) different input data. Think of this as an
                 ensemble of `n_estimators`-many "prompts" of the input data.

            categorical_features_indices:
                The indices of the columns that are suggested to be treated as
                categorical. If `None`, the model will infer the categorical columns.
                If provided, we might ignore some of the suggestion to better fit the
                data seen during pre-training.

                !!! note
                    The indices are 0-based and should represent the data passed to
                    `.fit()`. If the data changes between the initializations of the
                    model and the `.fit()`, consider setting the
                    `.categorical_features_indices` attribute after the model was
                    initialized and before `.fit()`.

            softmax_temperature:
                The temperature for the softmax function. This is used to control the
                confidence of the model's predictions. Lower values make the model's
                predictions more confident. This is only applied when predicting during
                a post-processing step. Set `softmax_temperature=1.0` for no effect.

            balance_probabilities:
                Whether to balance the probabilities based on the class distribution
                in the training data. This can help to improve predictive performance
                when the classes are highly imbalanced. This is only applied when
                predicting during a post-processing step.

            average_before_softmax:
                Only used if `n_estimators > 1`. Whether to average the predictions of
                the estimators before applying the softmax function. This can help to
                improve predictive performance when there are many classes or when
                calibrating the model's confidence. This is only applied when predicting
                during a post-processing.

                - If `True`, the predictions are averaged before applying the softmax
                  function. Thus, we average the logits of TabPFN and then apply the
                  softmax.
                - If `False`, the softmax function is applied to each set of logits.
                  Then, we average the resulting probabilities of each forward pass.

            model_path:
                The path to the TabPFN model file, i.e., the pre-trained weights.

                - If `"auto"`, the model will be downloaded upon first use. This
                  defaults to your system cache directory, but can be overwritten
                  with the use of an environment variable `TABPFN_MODEL_CACHE_DIR`.
                - If a path or a string of a path, the model will be loaded from
                  the user-specified location if available, otherwise it will be
                  downloaded to this location.

            device:
                The device to use for inference with TabPFN. If `"auto"`, the device
                is `"cuda"` if available, otherwise `"cpu"`.

                See PyTorch's documentation on devices for more information about
                supported devices.


            ignore_pretraining_limits:
                Whether to ignore the pre-training limits of the model. The TabPFN
                models have been pre-trained on a specific range of input data. If the
                input data is outside of this range, the model may not perform well.
                You may ignore our limits to use the model on data outside the
                pre-training range.

                - If `True`, the model will not raise an error if the input data is
                  outside the pre-training range.
                - If `False`, you can use the model outside the pre-training range, but
                  the model could perform worse.

                !!! note

                    The current pre-training limits are:

                    - 10_000 samples/rows
                    - 500 features/columns
                    - 10 classes, this is not ignorable and will raise an error
                      if the model is used with more classes.

            inference_precision:
                The precision to use for inference. This can dramatically affect the
                speed and reproducibility of the inference. Higher precision can lead to
                better reproducibility but at the cost of speed. By default, we optimize
                for speed and use torch's mixed-precision autocast. The options are:

                - If `torch.dtype`, we force precision of the model and data to be
                  the specified torch.dtype during inference. This can is particularly
                  useful for reproducibility. Here, we do not use mixed-precision.
                - If `"autocast"`, enable PyTorch's mixed-precision autocast. Ensure
                  that your device is compatible with mixed-precision.
                - If `"auto"`, we determine whether to use autocast or not depending on
                  the device type.

            fit_mode:
                Determine how the TabPFN model is "fitted". The mode determines how the
                data is preprocessed and cached for inference. This is unique to an
                in-context learning foundation model like TabPFN, as the "fitting" is
                technically the
                forward pass of the model. The options are:

                - If `"low_memory"`, the data is preprocessed on-demand during inference
                  when calling `.predict()` or `.predict_proba()`. This is the most
                  memory-efficient mode but can be slower for large datasets because
                  the data is (repeatedly) preprocessed on-the-fly.
                  Ideal with low GPU memory and/or a single call to `.fit()` and
                  `.predict()`.
                - If `"fit_preprocessors"`, the data is preprocessed and cached once
                  during the `.fit()` call. During inference, the cached preprocessing
                  (of the training data) is used instead of re-computing it.
                  Ideal with low GPU memory and multiple calls to `.predict()` with
                  the same training data.
                - If `"fit_with_cache"`, the data is preprocessed and cached once during
                  the `.fit()` call like in `fit_preprocessors`. Moreover, the
                  transformer key-value cache is also initialized, allowing for much
                  faster inference on the same data at a large cost of memory.
                  Ideal with very high GPU memory and multiple calls to `.predict()`
                  with the same training data.

            memory_saving_mode:
                Enable GPU/CPU memory saving mode. This can help to prevent
                out-of-memory errors that result from computations that would consume
                more memory than available on the current device. We save memory by
                automatically batching certain model computations within TabPFN to
                reduce the total required memory. The options are:

                - If `bool`, enable/disable memory saving mode.
                - If `"auto"`, we will estimate the amount of memory required for the
                  forward pass and apply memory saving if it is more than the
                  available GPU/CPU memory. This is the recommended setting as it
                  allows for speed-ups and prevents memory errors depending on
                  the input data.
                - If `float` or `int`, we treat this value as the maximum amount of
                  available GPU/CPU memory (in GB). We will estimate the amount
                  of memory required for the forward pass and apply memory saving
                  if it is more than this value. Passing a float or int value for
                  this parameter is the same as setting it to True and explicitly
                  specifying the maximum free available memory.

                !!! warning
                    This does not batch the original input data. We still recommend to
                    batch this as necessary if you run into memory errors! For example,
                    if the entire input data does not fit into memory, even the memory
                    save mode will not prevent memory errors.

            random_state:
                Controls the randomness of the model. Pass an int for reproducible
                results and see the scikit-learn glossary for more information. If
                `None`, the randomness is determined by the system when calling
                `.fit()`.

                !!! warning
                    We depart from the usual scikit-learn behavior in that by default
                    we provide a fixed seed of `0`.

                !!! note
                    Even if a seed is passed, we cannot always guarantee reproducibility
                    due to PyTorch's non-deterministic operations and general numerical
                    instability. To get the most reproducible results across hardware,
                    we recommend using a higher precision as well (at the cost of a
                    much higher inference time). Likewise, for scikit-learn, consider
                    passing `USE_SKLEARN_16_DECIMAL_PRECISION=True` as kwarg.

            n_jobs:
                The number of workers for tasks that can be parallelized across CPU
                cores. Currently, this is used for preprocessing the data in parallel
                (if `n_estimators > 1`).

                - If `-1`, all available CPU cores are used.
                - If `int`, the number of CPU cores to use is determined by `n_jobs`.

            inference_config:
                For advanced users, additional advanced arguments that adjust the
                behavior of the model interface.
                See [tabpfn.constants.ModelInterfaceConfig][] for details and options.

                - If `None`, the default ModelInterfaceConfig is used.
                - If `dict`, the key-value pairs are used to update the default
                  `ModelInterfaceConfig`. Raises an error if an unknown key is passed.
                - If `ModelInterfaceConfig`, the object is used as the configuration.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.model_path = model_path
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision: torch.dtype | Literal["autocast", "auto"] = (
            inference_precision
        )
        self.fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"] = (
            fit_mode
        )
        self.memory_saving_mode: bool | Literal["auto"] | float | int = (
            memory_saving_mode
        )
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.inference_config = inference_config

    # TODO: We can remove this from scikit-learn lower bound of 1.6
    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
            "multilabel": False,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()  # type: ignore
        tags.input_tags.allow_nan = True
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X: XType, y: YType,sample_size:int=10000) -> Self:
        """Fit the model.

        Args:
            X: The input data.
            y: The target variable.
        """
        static_seed, rng = infer_random_state(self.random_state)

        # Load the model and config
        self.model_, self.config_, _ = initialize_tabpfn_model(
            model_path=self.model_path,
            which="classifier",
            fit_mode=self.fit_mode,
            static_seed=static_seed,
        )

        # Determine device and precision
        self.device_ = infer_device_and_type(self.device)
        (self.use_autocast_, self.forced_inference_dtype_, byte_size) = (
            determine_precision(self.inference_precision, self.device_)
        )

        # Build the interface_config
        self.interface_config_ = ModelInterfaceConfig.from_user_input(
            inference_config=self.inference_config,
        )

        outlier_removal_std = self.interface_config_.OUTLIER_REMOVAL_STD
        if outlier_removal_std == "auto":
            outlier_removal_std = (
                self.interface_config_._CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD
            )
        update_encoder_outlier_params(
            model=self.model_,
            remove_outliers_std=outlier_removal_std,
            seed=static_seed,
            inplace=True,
        )

        X, y, feature_names_in, n_features_in = validate_Xy_fit(
            X,
            y,
            estimator=self,
            ensure_y_numeric=False,
            max_num_samples=self.interface_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.interface_config_.MAX_NUMBER_OF_FEATURES,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )
        if feature_names_in is not None:
            self.feature_names_in_ = feature_names_in
        self.n_features_in_ = n_features_in

        # Ensure that the y values are ordinally encoded
        # TODO(eddiebergman): Ensure the counts here line up with
        #   the actual classes after label encoder.
        _, counts = np.unique(y, return_counts=True)
        self.class_counts_ = counts
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_  # type: ignore
        self.n_classes_ = len(self.classes_)

        # TODO: Support more classes with a fallback strategy.
        if self.n_classes_ > self.interface_config_.MAX_NUMBER_OF_CLASSES:
            raise ValueError(
                f"Number of classes {self.n_classes_} exceeds the maximal number of "
                f"classes supported by TabPFN. Consider using a strategy to reduce "
                f"the number of classes (e.g., OneVsRest).",
            )

        # Will convert specified categorical indices to category dtype, as well
        # as handle `np.object` arrays or otherwise `object` dtype pandas columns.
        X = _fix_dtypes(X, cat_indices=self.categorical_features_indices)

        # Ensure categories are ordinally encoded
        ord_encoder = _get_ordinal_encoder()
        X = ord_encoder.fit_transform(X)  # type: ignore
        assert isinstance(X, np.ndarray)
        self.preprocessor_ = ord_encoder

        self.inferred_categorical_indices_ = infer_categorical_features(
            X=X,
            provided=self.categorical_features_indices,
            min_samples_for_inference=self.interface_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
            max_unique_for_category=self.interface_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
            min_unique_for_numerical=self.interface_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
        )

        # Now we build the ensemble configurations with the four main elements:
        #   feature_shifts, subsamples, class_perms, preprocessor_configs
        preprocess_transforms = self.interface_config_.PREPROCESS_TRANSFORMS
        ensemble_configs = EnsembleConfig.generate_for_classification(
            n=self.n_estimators,
            
            subsample_size = min(sample_size,len(X)),
            add_fingerprint_feature=self.interface_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.interface_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.interface_config_.POLYNOMIAL_FEATURES,
            max_index=len(X),
            preprocessor_configs=(
                preprocess_transforms
                if preprocess_transforms is not None
                else default_classifier_preprocessor_configs()
            ),
            class_shift_method=self.interface_config_.CLASS_SHIFT_METHOD,
            n_classes=self.n_classes_,
            random_state=rng,
        )
        assert len(ensemble_configs) == self.n_estimators

        # Create the inference engine
        self.executor_ = create_inference_engine(
            X_train=X,
            y_train=y,
            model=self.model_,
            ensemble_configs=ensemble_configs,
            cat_ix=self.inferred_categorical_indices_,
            fit_mode=self.fit_mode,
            device_=self.device_,
            rng=rng,
            n_jobs=self.n_jobs,
            byte_size=byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
        )

        return self

    def predict(self, X: XType) -> np.ndarray:
        """Predict the class labels for the provided input samples.

        Args:
            X: The input samples.

        Returns:
            The predicted class labels.
        """
        proba = self.predict_proba(X)
        y = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(y)  # type: ignore

    def predict_proba(self, X: XType) -> np.ndarray:
        """Predict the probabilities of the classes for the provided input samples.

        Args:
            X: The input data.

        Returns:
            The predicted probabilities of the classes.
        """
        check_is_fitted(self)

        X = validate_X_predict(X, self)
        X = _fix_dtypes(X, cat_indices=self.categorical_features_indices)
        X = self.preprocessor_.transform(X)

        outputs: list[torch.Tensor] = []

        for output, config in self.executor_.iter_outputs(
            X,
            device=self.device_,
            autocast=self.use_autocast_,
        ):
            assert isinstance(config, ClassifierEnsembleConfig)
            # Cut out logits for classes which do not exist
            assert output.ndim == 2

            if self.softmax_temperature != 1:
                output = (  # noqa: PLW2901
                    output[:, : self.n_classes_].float() / self.softmax_temperature
                )

            # Reverse class permutation if exists
            if config.class_permutation is not None:
                output = output[..., config.class_permutation]  # noqa: PLW2901

            outputs.append(output)

        if self.average_before_softmax:
            output = torch.stack(outputs).mean(dim=0)
            output = torch.nn.functional.softmax(output, dim=1)
        else:
            # Softmax each 2d outputs before average
            outputs = [torch.nn.functional.softmax(o, dim=1) for o in outputs]
            output = torch.stack(outputs).mean(dim=0)

        if self.balance_probabilities:
            class_prob_in_train = self.class_counts_ / self.class_counts_.sum()
            output = output * torch.Tensor(class_prob_in_train).to(self.device_)
            output = output / output.sum(dim=-1, keepdim=True)

        output = output.float().cpu().numpy()

        if self.interface_config_.USE_SKLEARN_16_DECIMAL_PRECISION:
            output = np.around(output, decimals=SKLEARN_16_DECIMAL_PRECISION)
            output = np.where(output < PROBABILITY_EPSILON_ROUND_ZERO, 0.0, output)

        # Normalize to guarantee proba sum to 1, required due to precision issues and
        # going from torch to numpy
        return output / output.sum(axis=1, keepdims=True)  # type: ignore

class TabPFNRegressor(RegressorMixin, BaseEstimator):
    """TabPFNRegressor class."""

    config_: InferenceConfig
    """The configuration of the loaded model to be used for inference."""

    interface_config_: ModelInterfaceConfig
    """Additional configuration of the interface for expert users."""

    device_: torch.device
    """The device determined to be used."""

    feature_names_in_: npt.NDArray[Any]
    """The feature names of the input data.

    May not be set if the input data does not have feature names,
    such as with a numpy array.
    """

    n_features_in_: int
    """The number of features in the input data used during `fit()`."""

    inferred_categorical_indices_: list[int]
    """The indices of the columns that were inferred to be categorical,
    as a product of any features deemed categorical by the user and what would
    work best for the model.
    """

    n_outputs_: Literal[1]  # We only support single output
    """The number of outputs the model supports. Only 1 for now"""

    bardist_: FullSupportBarDistribution
    """The bar distribution of the target variable, used by the model."""

    renormalized_criterion_: FullSupportBarDistribution
    """The normalized bar distribution used for computing the predictions."""

    use_autocast_: bool
    """Whether torch's autocast should be used."""

    forced_inference_dtype_: _dtype | None
    """The forced inference dtype for the model based on `inference_precision`."""

    executor_: InferenceEngine
    """The inference engine used to make predictions."""

    y_train_mean_: float
    """The mean of the target variable during training."""

    y_train_std: float
    """The standard deviation of the target variable during training."""

    preprocessor_: ColumnTransformer
    """The column transformer used to preprocess the input data to be numeric."""

    # TODO: consider moving the following to constants.py
    _OUTPUT_TYPES_BASIC = ("mean", "median", "mode")
    """The basic output types supported by the model."""
    _OUTPUT_TYPES_QUANTILES = ("quantiles",)
    """The quantiles output type supported by the model."""
    _OUTPUT_TYPES = _OUTPUT_TYPES_BASIC + _OUTPUT_TYPES_QUANTILES
    """The output types supported by the model for the "main" output type."""
    _OUTPUT_TYPES_COMPOSITE = ("full", "main")
    """The composite output types supported by the model."""
    _USABLE_OUTPUT_TYPES = _OUTPUT_TYPES + _OUTPUT_TYPES_COMPOSITE
    """The output types supported by the model."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_estimators: int = 8,
        categorical_features_indices: Sequence[int] | None = None,
        softmax_temperature: float = 0.9,
        average_before_softmax: bool = False,
        model_path: str | Path | Literal["auto"] = "auto",
        device: str | torch.device | Literal["auto"] = "auto",
        ignore_pretraining_limits: bool = False,
        inference_precision: _dtype | Literal["autocast", "auto"] = "auto",
        fit_mode: Literal[
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
        ] = "fit_preprocessors",
        memory_saving_mode: bool | Literal["auto"] | float | int = "auto",
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
        n_jobs: int = -1,
        inference_config: dict | ModelInterfaceConfig | None = None,
    ) -> None:
        """A TabPFN interface for regression.

        Args:
            n_estimators:
                The number of estimators in the TabPFN ensemble. We aggregate the
                predictions of `n_estimators`-many forward passes of TabPFN.
                Each forward pass has (slightly) different input data. Think of this
                as an ensemble of `n_estimators`-many "prompts" of the input data.

            categorical_features_indices:
                The indices of the columns that are suggested to be treated as
                categorical. If `None`, the model will infer the categorical columns.
                If provided, we might ignore some of the suggestion to better fit the
                data seen during pre-training.

                !!! note
                    The indices are 0-based and should represent the data passed to
                    `.fit()`. If the data changes between the initializations of the
                    model and the `.fit()`, consider setting the
                    `.categorical_features_indices` attribute after the model was
                    initialized and before `.fit()`.

            softmax_temperature:
                The temperature for the softmax function. This is used to control the
                confidence of the model's predictions. Lower values make the model's
                predictions more confident. This is only applied when predicting during
                a post-processing step. Set `softmax_temperature=1.0` for no effect.

            average_before_softmax:
                Only used if `n_estimators > 1`. Whether to average the predictions of
                the estimators before applying the softmax function. This can help to
                improve predictive performance when there are many classes or when
                calibrating the model's confidence. This is only applied when
                predicting during a post-processing.

                - If `True`, the predictions are averaged before applying the softmax
                  function. Thus, we average the logits of TabPFN and then apply the
                  softmax.
                - If `False`, the softmax function is applied to each set of logits.
                  Then, we average the resulting probabilities of each forward pass.

            model_path:
                The path to the TabPFN model file, i.e., the pre-trained weights.

                - If `"auto"`, the model will be downloaded upon first use. This
                  defaults to your system cache directory, but can be overwritten
                  with the use of an environment variable `TABPFN_MODEL_CACHE_DIR`.
                - If a path or a string of a path, the model will be loaded from
                  the user-specified location if available, otherwise it will be
                  downloaded to this location.

            ignore_pretraining_limits:
                Whether to ignore the pre-training limits of the model. The TabPFN
                models have been pre-trained on a specific range of input data. If the
                input data is outside of this range, the model may not perform well.
                You may ignore our limits to use the model on data outside the
                pre-training range.

                - If `True`, the model will not raise an error if the input data is
                  outside the pre-training range.
                - If `False`, you can use the model outside the pre-training range, but
                  the model could perform worse.

                !!! note

                    The current pre-training limits are:

                    - 10_000 samples/rows
                    - 500 features/columns

            device:
                The device to use for inference with TabPFN. If `"auto"`, the device is
                `"cuda"` if available, otherwise `"cpu"`.

                See PyTorch's documentation on devices for more information about
                supported devices.

            inference_precision:
                The precision to use for inference. This can dramatically affect the
                speed and reproducibility of the inference. Higher precision can lead to
                better reproducibility but at the cost of speed. By default, we optimize
                for speed and use torch's mixed-precision autocast. The options are:

                - If `torch.dtype`, we force precision of the model and data to be
                  the specified torch.dtype during inference. This can is particularly
                  useful for reproducibility. Here, we do not use mixed-precision.
                - If `"autocast"`, enable PyTorch's mixed-precision autocast. Ensure
                  that your device is compatible with mixed-precision.
                - If `"auto"`, we determine whether to use autocast or not depending on
                  the device type.

            fit_mode:
                Determine how the TabPFN model is "fitted". The mode determines how the
                data is preprocessed and cached for inference. This is unique to an
                in-context learning foundation model like TabPFN, as the "fitting" is
                technically the forward pass of the model. The options are:

                - If `"low_memory"`, the data is preprocessed on-demand during inference
                  when calling `.predict()` or `.predict_proba()`. This is the most
                  memory-efficient mode but can be slower for large datasets because
                  the data is (repeatedly) preprocessed on-the-fly.
                  Ideal with low GPU memory and/or a single call to `.fit()` and
                  `.predict()`.
                - If `"fit_preprocessors"`, the data is preprocessed and cached once
                  during the `.fit()` call. During inference, the cached preprocessing
                  (of the training data) is used instead of re-computing it.
                  Ideal with low GPU memory and multiple calls to `.predict()` with
                  the same training data.
                - If `"fit_with_cache"`, the data is preprocessed and cached once during
                  the `.fit()` call like in `fit_preprocessors`. Moreover, the
                  transformer key-value cache is also initialized, allowing for much
                  faster inference on the same data at a large cost of memory.
                  Ideal with very high GPU memory and multiple calls to `.predict()`
                  with the same training data.

            memory_saving_mode:
                Enable GPU/CPU memory saving mode. This can help to prevent
                out-of-memory errors that result from computations that would consume
                more memory than available on the current device. We save memory by
                automatically batching certain model computations within TabPFN to
                reduce the total required memory. The options are:

                - If `bool`, enable/disable memory saving mode.
                - If `"auto"`, we will estimate the amount of memory required for the
                  forward pass and apply memory saving if it is more than the
                  available GPU/CPU memory. This is the recommended setting as it
                  allows for speed-ups and prevents memory errors depending on
                  the input data.
                - If `float` or `int`, we treat this value as the maximum amount of
                  available GPU/CPU memory (in GB). We will estimate the amount
                  of memory required for the forward pass and apply memory saving
                  if it is more than this value. Passing a float or int value for
                  this parameter is the same as setting it to True and explicitly
                  specifying the maximum free available memory

                !!! warning
                    This does not batch the original input data. We still recommend to
                    batch this as necessary if you run into memory errors! For example,
                    if the entire input data does not fit into memory, even the memory
                    save mode will not prevent memory errors.

            random_state:
                Controls the randomness of the model. Pass an int for reproducible
                results and see the scikit-learn glossary for more information.
                If `None`, the randomness is determined by the system when calling
                `.fit()`.

                !!! warning
                    We depart from the usual scikit-learn behavior in that by default
                    we provide a fixed seed of `0`.

                !!! note
                    Even if a seed is passed, we cannot always guarantee reproducibility
                    due to PyTorch's non-deterministic operations and general numerical
                    instability. To get the most reproducible results across hardware,
                    we recommend using a higher precision as well (at the cost of a
                    much higher inference time). Likewise, for scikit-learn, consider
                    passing `USE_SKLEARN_16_DECIMAL_PRECISION=True` as kwarg.

            n_jobs:
                The number of workers for tasks that can be parallelized across CPU
                cores. Currently, this is used for preprocessing the data in parallel
                (if `n_estimators > 1`).

                - If `-1`, all available CPU cores are used.
                - If `int`, the number of CPU cores to use is determined by `n_jobs`.

            inference_config:
                For advanced users, additional advanced arguments that adjust the
                behavior of the model interface.
                See [tabpfn.constants.ModelInterfaceConfig][] for details and options.

                - If `None`, the default ModelInterfaceConfig is used.
                - If `dict`, the key-value pairs are used to update the default
                  `ModelInterfaceConfig`. Raises an error if an unknown key is passed.
                - If `ModelInterfaceConfig`, the object is used as the configuration.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.model_path = model_path
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision: torch.dtype | Literal["autocast", "auto"] = (
            inference_precision
        )
        self.fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"] = (
            fit_mode
        )
        self.memory_saving_mode: bool | Literal["auto"] | float | int = (
            memory_saving_mode
        )
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.inference_config = inference_config

    # TODO: We can remove this from scikit-learn lower bound of 1.6
    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "regressor"
        return tags

    def fit(self, X: XType, y: YType,sample_size:int=10000) -> Self:
        """Fit the model.

        Args:
            X: The input data.
            y: The target variable.

        Returns:
            self
        """
        static_seed, rng = infer_random_state(self.random_state)

        # Load the model and config
        self.model_, self.config_, self.bardist_ = initialize_tabpfn_model(
            model_path=self.model_path,
            which="regressor",
            fit_mode=self.fit_mode,
            static_seed=static_seed,
        )

        # Determine device and precision
        self.device_ = infer_device_and_type(self.device)
        (self.use_autocast_, self.forced_inference_dtype_, byte_size) = (
            determine_precision(self.inference_precision, self.device_)
        )

        # Build the interface_config
        self.interface_config_ = ModelInterfaceConfig.from_user_input(
            inference_config=self.inference_config,
        )

        outlier_removal_std = self.interface_config_.OUTLIER_REMOVAL_STD
        if outlier_removal_std == "auto":
            outlier_removal_std = (
                self.interface_config_._REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD
            )
        update_encoder_outlier_params(
            model=self.model_,
            remove_outliers_std=outlier_removal_std,
            seed=static_seed,
            inplace=True,
        )

        X, y, feature_names_in, n_features_in = validate_Xy_fit(
            X,
            y,
            estimator=self,
            ensure_y_numeric=False,
            max_num_samples=self.interface_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.interface_config_.MAX_NUMBER_OF_FEATURES,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )
        assert isinstance(X, np.ndarray)

        if feature_names_in is not None:
            self.feature_names_in_ = feature_names_in
        self.n_features_in_ = n_features_in

        # Will convert specified categorical indices to category dtype, as well
        # as handle `np.object` arrays or otherwise `object` dtype pandas columns.
        X = _fix_dtypes(X, cat_indices=self.categorical_features_indices)

        # Ensure categories are ordinally encoded
        ord_encoder = _get_ordinal_encoder()
        X = ord_encoder.fit_transform(X)  # type: ignore
        self.preprocessor_ = ord_encoder

        self.inferred_categorical_indices_ = infer_categorical_features(
            X=X,
            provided=self.categorical_features_indices,
            min_samples_for_inference=self.interface_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
            max_unique_for_category=self.interface_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
            min_unique_for_numerical=self.interface_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
        )

        possible_target_transforms = (
            ReshapeFeatureDistributionsStep.get_all_preprocessors(
                num_examples=y.shape[0],
                random_state=static_seed,
            )
        )
        target_preprocessors: list[TransformerMixin | Pipeline | None] = []
        for (
            y_target_preprocessor
        ) in self.interface_config_.REGRESSION_Y_PREPROCESS_TRANSFORMS:
            if y_target_preprocessor is not None:
                preprocessor = possible_target_transforms[y_target_preprocessor]
            else:
                preprocessor = None

            target_preprocessors.append(preprocessor)
        preprocess_transforms = self.interface_config_.PREPROCESS_TRANSFORMS

        ensemble_configs = EnsembleConfig.generate_for_regression(
            n=self.n_estimators,
            subsample_size=min(sample_size,len(X)),
            add_fingerprint_feature=self.interface_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.interface_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.interface_config_.POLYNOMIAL_FEATURES,
            max_index=len(X),
            preprocessor_configs=(
                preprocess_transforms
                if preprocess_transforms is not None
                else default_regressor_preprocessor_configs()
            ),
            target_transforms=target_preprocessors,
            random_state=rng,
        )
        assert len(ensemble_configs) == self.n_estimators

        # Standardize y
        mean = np.mean(y)
        std = np.std(y)
        self.y_train_std_ = std.item() + 1e-20
        self.y_train_mean_ = mean.item()
        y = (y - self.y_train_mean_) / self.y_train_std_
        self.renormalized_criterion_ = FullSupportBarDistribution(
            self.bardist_.borders * self.y_train_std_ + self.y_train_mean_,
        ).float()

        # Create the inference engine
        self.executor_ = create_inference_engine(
            X_train=X,
            y_train=y,
            model=self.model_,
            ensemble_configs=ensemble_configs,
            cat_ix=self.inferred_categorical_indices_,
            fit_mode=self.fit_mode,
            device_=self.device_,
            rng=rng,
            n_jobs=self.n_jobs,
            byte_size=byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
        )

        return self

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["mean", "median", "mode"] = "mean",
        quantiles: list[float] | None = None,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["quantiles"],
        quantiles: list[float] | None = None,
    ) -> list[np.ndarray]: ...

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["main"],
        quantiles: list[float] | None = None,
    ) -> dict[str, np.ndarray]: ...

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["full"],
        quantiles: list[float] | None = None,
    ) -> dict[str, np.ndarray | FullSupportBarDistribution]: ...

    # FIXME: improve to not have noqa C901, PLR0912
    def predict(  # noqa: C901, PLR0912
        self,
        X: XType,
        *,
        # TODO: support "ei", "pi"
        output_type: Literal[
            "mean",
            "median",
            "mode",
            "quantiles",
            "full",
            "main",
        ] = "mean",
        quantiles: list[float] | None = None,
    ) -> (
        np.ndarray
        | list[np.ndarray]
        | dict[str, np.ndarray]
        | dict[str, np.ndarray | FullSupportBarDistribution]
    ):
        """Predict the target variable.

        Args:
            X: The input data.
            output_type:
                Determines the type of output to return.

                - If `"mean"`, we return the mean over the predicted distribution.
                - If `"median"`, we return the median over the predicted distribution.
                - If `"mode"`, we return the mode over the predicted distribution.
                - If `"quantiles"`, we return the quantiles of the predicted
                    distribution. The parameter `output_quantiles` determines which
                    quantiles are returned.
                - If `"main"`, we return the all output types above in a dict.
                - If `"full"`, we return the full output of the model, including the
                  logits and the criterion, and all the output types from "main".

            quantiles:
                The quantiles to return if `output="quantiles"`.

                By default, the `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`
                quantiles are returned. The predictions per quantile match
                the input order.

        Returns:
            The predicted target variable or a list of predictions per quantile.
        """
        check_is_fitted(self)

        X = validate_X_predict(X, self)
        X = _fix_dtypes(X, cat_indices=self.categorical_features_indices)
        X = self.preprocessor_.transform(X)

        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            assert all(
                (0 <= q <= 1) and (isinstance(q, float)) for q in quantiles
            ), "All quantiles must be between 0 and 1 and floats."
        if output_type not in self._USABLE_OUTPUT_TYPES:
            raise ValueError(f"Invalid output type: {output_type}")

        std_borders = self.bardist_.borders.cpu().numpy()
        outputs: list[torch.Tensor] = []
        borders: list[np.ndarray] = []

        for output, config in self.executor_.iter_outputs(
            X,
            device=self.device_,
            autocast=self.use_autocast_,
        ):
            assert isinstance(config, RegressorEnsembleConfig)

            if self.softmax_temperature != 1:
                output = output.float() / self.softmax_temperature  # noqa: PLW2901

            borders_t: np.ndarray
            logit_cancel_mask: np.ndarray | None
            descending_borders: bool

            # TODO(eddiebergman): Maybe this could be parallelized or done in fit
            # but I somehow doubt it takes much time to be worth it.
            # One reason to make it worth it is if you want fast predictions, i.e.
            # don't re-do this each time.
            # However it gets a bit more difficult as you need to line up the
            # outputs from `iter_outputs` above (which may be in arbitrary order),
            # along with the specific config the output belongs to. This is because
            # the transformation done to the borders for a given output is dependant
            # upon the target_transform of the config.
            if config.target_transform is None:
                borders_t = std_borders.copy()
                logit_cancel_mask = None
                descending_borders = False
            else:
                logit_cancel_mask, descending_borders, borders_t = (
                    _transform_borders_one(
                        std_borders,
                        target_transform=config.target_transform,
                        repair_nan_borders_after_transform=self.interface_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM,
                    )
                )
                if descending_borders:
                    borders_t = borders_t.flip(-1)  # type: ignore

            borders.append(borders_t)

            if logit_cancel_mask is not None:
                output = output.clone()  # noqa: PLW2901
                output[..., logit_cancel_mask] = float("-inf")

            outputs.append(output)  # type: ignore

        transformed_logits = [
            translate_probs_across_borders(
                logits,
                frm=torch.as_tensor(borders_t, device=self.device_),
                to=self.bardist_.borders.to(self.device_),
            )
            for logits, borders_t in zip(outputs, borders)
        ]
        stacked_logits = torch.stack(transformed_logits, dim=0)
        if self.average_before_softmax:
            logits = stacked_logits.log().mean(dim=0).softmax(dim=-1)
        else:
            logits = stacked_logits.mean(dim=0)

        # Post-process the logits
        logits = logits.log()
        if logits.dtype == torch.float16:
            logits = logits.float()
        logits = logits.cpu()

        # Determine and return intended output type
        logit_to_output = partial(
            _logits_to_output,
            logits=logits,
            criterion=self.renormalized_criterion_,
            quantiles=quantiles,
        )
        if output_type in ["full", "main"]:
            output = {k: logit_to_output(output_type=k) for k in self._OUTPUT_TYPES}

            if output_type == "full":
                output = {
                    "criterion": self.renormalized_criterion_,
                    "logits": logits,
                    **output,
                }
            return output  # type: ignore

        return logit_to_output(output_type=output_type)


def _logits_to_output(
    *,
    output_type: str,
    logits: torch.Tensor,
    criterion: FullSupportBarDistribution,
    quantiles: list[float],
) -> np.ndarray | list[np.ndarray]:
    """Convert the logits to the specified output type.

    Args:
        output_type: The output type to convert the logits to.
        logits: The logits to convert.
        criterion: The criterion to use for the conversion.
        quantiles: The quantiles to use for the conversion.

    Returns:
        The converted logits or list of converted logits.
    """
    if output_type == "quantiles":
        return [criterion.icdf(logits, q).cpu().detach().numpy() for q in quantiles]

    # TODO: support
    #   "pi": criterion.pi(logits, np.max(self.y)), # noqa: ERA001
    #   "ei": criterion.ei(logits), # noqa: ERA001
    if output_type == "mean":
        output = criterion.mean(logits)
    elif output_type == "median":
        output = criterion.median(logits)
    elif output_type == "mode":
        output = criterion.mode(logits)
    else:
        raise ValueError(f"Invalid output type: {output_type}")

    return output.cpu().detach().numpy()  # type: ignore