# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Modified from Ultralytics kalman_filter.py
# Replaces the standard Kalman filter motion model with an Unscented Kalman Filter using nonlinear velocity damping for bounding box motion
# Modified in 2026


from __future__ import annotations

import numpy as np


class UKF:
    state_dim = 8
    measurement_dim = 4

    def __init__(
        self,
        dt: float = 1.0,
        alpha: float = 0.3,
        beta: float = 2.0,
        kappa: float = 0.0,
        process_noise_scale: float = 1.0,
        measurement_noise_scale: float = 1.0,
        min_box_size: float = 1e-3,
        drag: float = 0.08,
        size_drag: float | None = None,
    ):
        self.dt = float(dt)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        self.process_noise_scale = float(process_noise_scale)
        self.measurement_noise_scale = float(
            measurement_noise_scale
        )

        self.min_box_size = float(min_box_size)

        self.drag = float(drag)
        self.size_drag = (
            2.0 * self.drag
            if size_drag is None
            else float(size_drag)
        )

        if self.dt <= 0:
            raise ValueError("dt must be > 0")

        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")

        if self.beta < 0:
            raise ValueError("beta must be >= 0")

        if self.kappa < 0:
            raise ValueError("kappa must be >= 0")

        if self.process_noise_scale < 0:
            raise ValueError(
                "process_noise_scale must be >= 0"
            )

        if self.measurement_noise_scale < 0:
            raise ValueError(
                "measurement_noise_scale must be >= 0"
            )

        if self.min_box_size <= 0:
            raise ValueError(
                "min_box_size must be > 0"
            )

        if self.drag < 0:
            raise ValueError("drag must be >= 0")

        if self.size_drag < 0:
            raise ValueError(
                "size_drag must be >= 0"
            )

        n = self.state_dim

        self.lambda_ = (
            self.alpha**2 * (n + self.kappa)
            - n
        )

        n_plus_lambda = n + self.lambda_

        if n_plus_lambda <= 0:
            raise ValueError(
                "Invalid UKF parameters: "
                "state_dim + lambda must be > 0."
            )

        self.gamma = np.sqrt(n_plus_lambda)

        n_sigma = 2 * n + 1

        self.weights_mean = np.full(
            n_sigma,
            1.0 / (2.0 * n_plus_lambda),
            dtype=np.float64,
        )

        self.weights_mean[0] = (
            self.lambda_ / n_plus_lambda
        )

        self.weights_cov = self.weights_mean.copy()

        self.weights_cov[0] = (
            self.lambda_ / n_plus_lambda
            + (1.0 - self.alpha**2 + self.beta)
        )

        self._motion_std_position = 1.0 / 20.0
        self._motion_std_velocity = 1.0 / 160.0

    def constrain_state(
        self,
        state: np.ndarray,
    ) -> np.ndarray:

        state = np.asarray(
            state,
            dtype=np.float64,
        ).copy()

        if state.shape[-1] != self.state_dim:
            raise ValueError(
                "Expected final state dimension 8, "
                f"got {state.shape}."
            )

        state[..., 2:4] = np.maximum(
            state[..., 2:4],
            self.min_box_size,
        )

        return state

    def transition(
        self,
        state: np.ndarray,
    ) -> np.ndarray:

        state = np.asarray(
            state,
            dtype=np.float64,
        )

        if state.shape != (self.state_dim,):
            raise ValueError(
                "Expected state shape (8,), "
                f"got {state.shape}."
            )

        if not np.isfinite(state).all():
            raise ValueError(
                "State contains non-finite values."
            )

        out = state.copy()

        vx = state[4]
        vy = state[5]
        vw = state[6]
        vh = state[7]

        speed_xy = np.hypot(vx, vy)

        denominator_xy = (
            1.0
            + self.drag
            * speed_xy
            * self.dt
        )

        vx_next = vx / denominator_xy
        vy_next = vy / denominator_xy

        speed_size = np.hypot(vw, vh)

        denominator_size = (
            1.0
            + self.size_drag
            * speed_size
            * self.dt
        )

        vw_next = vw / denominator_size
        vh_next = vh / denominator_size

        out[0] = state[0] + vx_next * self.dt
        out[1] = state[1] + vy_next * self.dt
        out[2] = state[2] + vw_next * self.dt
        out[3] = state[3] + vh_next * self.dt

        out[4] = vx_next
        out[5] = vy_next
        out[6] = vw_next
        out[7] = vh_next

        return out

    def _transition_batch(
        self,
        states: np.ndarray,
    ) -> np.ndarray:

        states = np.asarray(
            states,
            dtype=np.float64,
        )

        if (
            states.ndim != 2
            or states.shape[1] != self.state_dim
        ):
            raise ValueError(
                "Expected states shape (N, 8), "
                f"got {states.shape}."
            )

        if not np.isfinite(states).all():
            raise ValueError(
                "States contain non-finite values."
            )

        out = states.copy()

        vx = states[:, 4]
        vy = states[:, 5]
        vw = states[:, 6]
        vh = states[:, 7]

        speed_xy = np.hypot(vx, vy)

        denominator_xy = (
            1.0
            + self.drag
            * speed_xy
            * self.dt
        )

        vx_next = vx / denominator_xy
        vy_next = vy / denominator_xy

        speed_size = np.hypot(vw, vh)

        denominator_size = (
            1.0
            + self.size_drag
            * speed_size
            * self.dt
        )

        vw_next = vw / denominator_size
        vh_next = vh / denominator_size

        out[:, 0] = (
            states[:, 0]
            + vx_next * self.dt
        )

        out[:, 1] = (
            states[:, 1]
            + vy_next * self.dt
        )

        out[:, 2] = (
            states[:, 2]
            + vw_next * self.dt
        )

        out[:, 3] = (
            states[:, 3]
            + vh_next * self.dt
        )

        out[:, 4] = vx_next
        out[:, 5] = vy_next
        out[:, 6] = vw_next
        out[:, 7] = vh_next

        return out

    @staticmethod
    def measurement_function(
        state: np.ndarray,
    ) -> np.ndarray:

        return np.asarray(
            state[..., :4],
            dtype=np.float64,
        )

    def _process_covariance(
        self,
        mean: np.ndarray,
    ) -> np.ndarray:

        w = max(
            float(mean[2]),
            self.min_box_size,
        )

        h = max(
            float(mean[3]),
            self.min_box_size,
        )

        std_pos = np.array(
            [
                self._motion_std_position * w,
                self._motion_std_position * h,
                self._motion_std_position * w,
                self._motion_std_position * h,
            ],
            dtype=np.float64,
        )

        std_vel = np.array(
            [
                self._motion_std_velocity * w,
                self._motion_std_velocity * h,
                self._motion_std_velocity * w,
                self._motion_std_velocity * h,
            ],
            dtype=np.float64,
        )

        std = np.concatenate(
            [std_pos, std_vel]
        )

        return (
            self.process_noise_scale
            * np.diag(std**2)
        )

    def _measurement_covariance(
        self,
        mean: np.ndarray,
        confidence: float | None = None,
    ) -> np.ndarray:

        w = max(
            float(mean[2]),
            self.min_box_size,
        )

        h = max(
            float(mean[3]),
            self.min_box_size,
        )

        std = np.array(
            [
                self._motion_std_position * w,
                self._motion_std_position * h,
                self._motion_std_position * w,
                self._motion_std_position * h,
            ],
            dtype=np.float64,
        )

        scale = self.measurement_noise_scale

        if confidence is not None:
            confidence = float(confidence)

            if not np.isfinite(confidence):
                raise ValueError(
                    "confidence must be finite."
                )

            scale *= max(
                1.0 - confidence,
                0.05,
            )

        return scale * np.diag(std**2)

    @staticmethod
    def _symmetrize(
        covariance: np.ndarray,
        jitter: float = 1e-9,
    ) -> np.ndarray:

        covariance = np.asarray(
            covariance,
            dtype=np.float64,
        )

        covariance = 0.5 * (
            covariance + covariance.T
        )

        if not np.isfinite(covariance).all():
            raise FloatingPointError(
                "Covariance contains non-finite values."
            )

        covariance = covariance + (
            jitter
            * np.eye(
                covariance.shape[0],
                dtype=np.float64,
            )
        )

        return covariance

    @staticmethod
    def _safe_cholesky(
        covariance: np.ndarray,
    ) -> np.ndarray:

        covariance = np.asarray(
            covariance,
            dtype=np.float64,
        )

        covariance = 0.5 * (
            covariance + covariance.T
        )

        if not np.isfinite(covariance).all():
            raise FloatingPointError(
                "Covariance contains non-finite values."
            )

        eye = np.eye(
            covariance.shape[0],
            dtype=np.float64,
        )

        # First try without adding unnecessary jitter
        jitter = 0.0

        for _ in range(10):
            try:
                return np.linalg.cholesky(
                    covariance + jitter * eye
                )
            except np.linalg.LinAlgError:
                jitter = (
                    1e-10
                    if jitter == 0.0
                    else jitter * 10.0
                )

        raise np.linalg.LinAlgError(
            "Covariance is not numerically "
            "positive definite."
        )

    def _sigma_points(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:

        mean = np.asarray(
            mean,
            dtype=np.float64,
        )

        covariance = np.asarray(
            covariance,
            dtype=np.float64,
        )

        if mean.shape != (self.state_dim,):
            raise ValueError(
                "Expected mean shape (8,), "
                f"got {mean.shape}."
            )

        if covariance.shape != (
            self.state_dim,
            self.state_dim,
        ):
            raise ValueError(
                "Expected covariance shape (8, 8), "
                f"got {covariance.shape}."
            )

        covariance = self._symmetrize(
            covariance
        )

        chol = self._safe_cholesky(
            covariance
        )

        n = self.state_dim

        points = np.empty(
            (
                2 * n + 1,
                n,
            ),
            dtype=np.float64,
        )

        points[0] = mean

        scaled_chol = (
            self.gamma * chol
        )

        # np.linalg.cholesky returns L where
        # P = L @ L.T.
        #
        # The columns of gamma*L provide the standard
        # covariance square-root directions.

        points[1 : n + 1] = (
            mean[None, :]
            + scaled_chol.T
        )

        points[n + 1 :] = (
            mean[None, :]
            - scaled_chol.T
        )

        return points

    def initiate(
        self,
        measurement: np.ndarray,
    ):
        """Create a track from an XYWH measurement.

        This preserves the Ultralytics-compatible contract:

            measurement -> (mean[8], covariance[8,8])
        """

        measurement = np.asarray(
            measurement,
            dtype=np.float64,
        )

        if measurement.shape != (4,):
            raise ValueError(
                "Expected measurement shape (4,), "
                f"got {measurement.shape}."
            )

        if not np.isfinite(
            measurement
        ).all():
            raise ValueError(
                "Measurement contains non-finite values."
            )

        measurement = measurement.copy()

        measurement[2:4] = np.maximum(
            measurement[2:4],
            self.min_box_size,
        )

        mean = np.zeros(
            self.state_dim,
            dtype=np.float64,
        )

        mean[:4] = measurement

        w = mean[2]
        h = mean[3]

        std = np.array(
            [
                2.0
                * self._motion_std_position
                * w,

                2.0
                * self._motion_std_position
                * h,

                2.0
                * self._motion_std_position
                * w,

                2.0
                * self._motion_std_position
                * h,

                10.0
                * self._motion_std_velocity
                * w,

                10.0
                * self._motion_std_velocity
                * h,

                10.0
                * self._motion_std_velocity
                * w,

                10.0
                * self._motion_std_velocity
                * h,
            ],
            dtype=np.float64,
        )

        covariance = np.diag(
            std**2
        )

        return (
            mean,
            covariance,
        )

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ):

        mean = np.asarray(
            mean,
            dtype=np.float64,
        )

        covariance = np.asarray(
            covariance,
            dtype=np.float64,
        )

        if mean.shape != (self.state_dim,):
            raise ValueError(
                "Expected mean shape (8,), "
                f"got {mean.shape}."
            )

        if covariance.shape != (
            self.state_dim,
            self.state_dim,
        ):
            raise ValueError(
                "Expected covariance shape (8, 8), "
                f"got {covariance.shape}."
            )

        if not np.isfinite(mean).all():
            raise ValueError(
                "Mean contains non-finite values."
            )

        if not np.isfinite(
            covariance
        ).all():
            raise ValueError(
                "Covariance contains non-finite values."
            )

        sigma_points = self._sigma_points(
            mean,
            covariance,
        )

        propagated = self._transition_batch(
            sigma_points
        )

        predicted_mean = (
            np.sum(
                self.weights_mean[:, None]
                * propagated,
                axis=0,
            )
        )

        # Do not clip sigma points, but constrain the actual
        # reconstructed state estimate.
        predicted_mean = (
            self.constrain_state(
                predicted_mean
            )
        )

        deltas = (
            propagated
            - predicted_mean[None, :]
        )

        predicted_covariance = (
            deltas.T
            @ (
                self.weights_cov[:, None]
                * deltas
            )
        )

        predicted_covariance += (
            self._process_covariance(
                predicted_mean
            )
        )

        predicted_covariance = (
            self._symmetrize(
                predicted_covariance
            )
        )

        return (
            predicted_mean,
            predicted_covariance,
        )

    def multi_predict(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
    ):

        means = np.asarray(
            means,
            dtype=np.float64,
        )

        covariances = np.asarray(
            covariances,
            dtype=np.float64,
        )

        if (
            means.ndim != 2
            or means.shape[1] != self.state_dim
        ):
            raise ValueError(
                "Expected means shape (N, 8), "
                f"got {means.shape}."
            )

        if (
            covariances.ndim != 3
            or covariances.shape[1:]
            != (
                self.state_dim,
                self.state_dim,
            )
        ):
            raise ValueError(
                "Expected covariances shape "
                "(N, 8, 8), "
                f"got {covariances.shape}."
            )

        if (
            means.shape[0]
            != covariances.shape[0]
        ):
            raise ValueError(
                "Means and covariances must "
                "contain the same number of tracks."
            )

        n_tracks = means.shape[0]

        if n_tracks == 0:
            return (
                means.copy(),
                covariances.copy(),
            )

        n = self.state_dim
        n_sigma = 2 * n + 1

        symmetric_covariances = (
            0.5
            * (
                covariances
                + np.swapaxes(
                    covariances,
                    1,
                    2,
                )
            )
        )

        sigma_points = np.empty(
            (
                n_tracks,
                n_sigma,
                n,
            ),
            dtype=np.float64,
        )

        sigma_points[:, 0, :] = means

        chol = np.empty_like(
            symmetric_covariances
        )

        for i in range(n_tracks):
            chol[i] = self._safe_cholesky(
                symmetric_covariances[i]
            )

        scaled_chol = (
            self.gamma * chol
        )

        sigma_points[
            :, 1 : n + 1, :
        ] = np.transpose(
            means[:, None, :]
            + np.transpose(
                scaled_chol,
                (0, 2, 1),
            ),
            (0, 1, 2),
        )

        sigma_points[
            :, n + 1 :, :
        ] = np.transpose(
            means[:, None, :]
            - np.transpose(
                scaled_chol,
                (0, 2, 1),
            ),
            (0, 1, 2),
        )

        flat_sigma_points = (
            sigma_points.reshape(
                n_tracks * n_sigma,
                n,
            )
        )

        propagated_flat = (
            self._transition_batch(
                flat_sigma_points
            )
        )

        propagated = (
            propagated_flat.reshape(
                n_tracks,
                n_sigma,
                n,
            )
        )

        predicted_means = np.sum(
            self.weights_mean[None, :, None]
            * propagated,
            axis=1,
        )

        predicted_means = (
            self.constrain_state(
                predicted_means
            )
        )

        deltas = (
            propagated
            - predicted_means[:, None, :]
        )

        predicted_covariances = np.einsum(
            "s,nsi,nsj->nij",
            self.weights_cov,
            deltas,
            deltas,
        )

        for i in range(n_tracks):
            predicted_covariances[i] += (
                self._process_covariance(
                    predicted_means[i]
                )
            )

        predicted_covariances = (
            0.5
            * (
                predicted_covariances
                + np.swapaxes(
                    predicted_covariances,
                    1,
                    2,
                )
            )
        )

        predicted_covariances += (
            1e-9
            * np.eye(
                n,
                dtype=np.float64,
            )[None, :, :]
        )

        return (
            predicted_means,
            predicted_covariances,
        )

    def _measurement_projection(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        confidence: float | None = None,
    ):

        mean = np.asarray(
            mean,
            dtype=np.float64,
        )

        covariance = np.asarray(
            covariance,
            dtype=np.float64,
        )

        sigma_points = self._sigma_points(
            mean,
            covariance,
        )

        z_points = (
            sigma_points[:, :4]
        )

        z_mean = np.sum(
            self.weights_mean[:, None]
            * z_points,
            axis=0,
        )

        z_mean[2:4] = np.maximum(
            z_mean[2:4],
            self.min_box_size,
        )

        z_deltas = (
            z_points
            - z_mean[None, :]
        )

        x_deltas = (
            sigma_points
            - mean[None, :]
        )

        z_covariance = np.einsum(
            "s,si,sj->ij",
            self.weights_cov,
            z_deltas,
            z_deltas,
        )

        state_cross_covariance = np.einsum(
            "s,si,sj->ij",
            self.weights_cov,
            x_deltas,
            z_deltas,
        )

        z_covariance += (
            self._measurement_covariance(
                mean,
                confidence,
            )
        )

        z_covariance = (
            self._symmetrize(
                z_covariance
            )
        )

        return (
            z_mean,
            z_covariance,
            state_cross_covariance,
        )

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        confidence: float | None = None,
    ):

        z_mean, z_covariance, _ = (
            self._measurement_projection(
                mean,
                covariance,
                confidence,
            )
        )

        return (
            z_mean,
            z_covariance,
        )

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float | None = None,
    ):

        mean = np.asarray(
            mean,
            dtype=np.float64,
        )

        covariance = np.asarray(
            covariance,
            dtype=np.float64,
        )

        measurement = np.asarray(
            measurement,
            dtype=np.float64,
        )

        if mean.shape != (self.state_dim,):
            raise ValueError(
                "Expected mean shape (8,), "
                f"got {mean.shape}."
            )

        if covariance.shape != (
            self.state_dim,
            self.state_dim,
        ):
            raise ValueError(
                "Expected covariance shape (8, 8), "
                f"got {covariance.shape}."
            )

        if measurement.shape != (
            self.measurement_dim,
        ):
            raise ValueError(
                "Expected measurement shape (4,), "
                f"got {measurement.shape}."
            )

        if not np.isfinite(
            measurement
        ).all():
            raise ValueError(
                "Measurement contains non-finite values."
            )

        measurement = measurement.copy()

        measurement[2:4] = np.maximum(
            measurement[2:4],
            self.min_box_size,
        )

        mean = self.constrain_state(
            mean
        )

        covariance = self._symmetrize(
            covariance
        )

        (
            z_mean,
            z_covariance,
            state_cross_covariance,
        ) = self._measurement_projection(
            mean,
            covariance,
            confidence,
        )

        kalman_gain = np.linalg.solve(
            z_covariance,
            state_cross_covariance.T,
        ).T

        innovation = (
            measurement - z_mean
        )

        new_mean = (
            mean
            + kalman_gain @ innovation
        )

        new_mean = self.constrain_state(
            new_mean
        )

        new_covariance = (
            covariance
            - kalman_gain
            @ z_covariance
            @ kalman_gain.T
        )

        new_covariance = (
            self._symmetrize(
                new_covariance
            )
        )

        return (
            new_mean,
            new_covariance,
        )

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:

        measurements = np.asarray(
            measurements,
            dtype=np.float64,
        )

        if (
            measurements.ndim != 2
            or measurements.shape[1]
            != self.measurement_dim
        ):
            raise ValueError(
                "Expected measurements shape "
                "(N, 4), "
                f"got {measurements.shape}."
            )

        if not np.isfinite(
            measurements
        ).all():
            raise ValueError(
                "Measurements contain non-finite values."
            )

        projected_mean, projected_covariance = (
            self.project(
                mean,
                covariance,
            )
        )

        if only_position:
            projected_mean = (
                projected_mean[:2]
            )

            projected_covariance = (
                projected_covariance[
                    :2,
                    :2,
                ]
            )

            measurements = (
                measurements[:, :2]
            )

        delta = (
            measurements
            - projected_mean
        )

        if metric == "gaussian":
            return np.sum(
                delta * delta,
                axis=1,
            )

        if metric == "maha":
            chol = self._safe_cholesky(
                projected_covariance
            )

            solved = np.linalg.solve(
                chol,
                delta.T,
            )

            return np.sum(
                solved * solved,
                axis=0,
            )

        raise ValueError(
            f"Invalid distance metric: {metric}"
        )