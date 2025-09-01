import time
from pathlib import Path

import torch
from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
from torch import Tensor


class SubmissionAv2:
    def __init__(self, save_dir: str = "") -> None:
        stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.submission_file = Path(save_dir) / f"single_agent_{stamp}.parquet"
        self.challenge_submission = ChallengeSubmission(predictions={})

    def format_data(
        self,
        data: dict,
        trajectory: Tensor,
        probability: Tensor,
        normalized_probability=False,
        inference=False,
    ) -> None:
        """
        trajectory: (B, A, M, 60, 2) - Batch, Agents, Modes, Timesteps, Coordinates
        probability: (B, A, M) - Batch, Agents, Modes
        """
        scenario_ids = data["scenario_id"]
        track_ids = data["track_id"]  # (B, A) - track_ids de todos los agentes
        batch_size = trajectory.shape[0]
        num_agents = trajectory.shape[1]

        # USAR SOLO LAS COORDENADAS DEL AGENTE FOCAL
        # Todos los datos ya están normalizados respecto al agente focal
        focal_origin = data["origin"].double()  # (B, 2) - origen del agente focal
        focal_theta = data["theta"].double()    # (B,) - orientación del agente focal
        
        # Expandir para aplicar la misma transformación a todos los agentes
        origin = focal_origin.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, 2)
        theta = focal_theta.unsqueeze(1)  # (B, 1)

        # Crear matriz de rotación INVERSA (para volver a coordenadas globales)
        # Nota: usamos -theta porque queremos la transformación inversa
        rotate_mat = torch.stack(
            [
                torch.cos(-theta),
                torch.sin(-theta),
                -torch.sin(-theta),
                torch.cos(-theta),
            ],
            dim=-1,
        ).reshape(batch_size, 1, 2, 2)  # (B, 1, 2, 2)

        with torch.no_grad():
            # Transformación INVERSA: de coordenadas normalizadas a globales
            # 1. Primero rotamos (deshacemos la rotación de normalización)
            rotated_trajectory = torch.matmul(
                trajectory[..., :2].double(), 
                rotate_mat.expand(-1, num_agents, -1, -1).unsqueeze(2)  # (B, A, 1, 2, 2)
            )  # (B, A, M, 60, 2)
            
            # 2. Luego trasladamos (deshacemos la traslación de normalización)
            global_trajectory = rotated_trajectory + origin  # (B, A, M, 60, 2)
            
            if not normalized_probability:
                probability = torch.softmax(probability.double(), dim=-1)

        global_trajectory = global_trajectory.detach().cpu().numpy()
        probability = probability.detach().cpu().numpy()

        # print(f"Global trajectory shape: {global_trajectory.shape}")
        # print(f"Global trajectory sample: {global_trajectory[0, :3, 0, :3, :2]}")  # Debugging line
        # AA

        if inference:
            return global_trajectory, probability

        # Guardar predicciones para cada agente en cada escenario
        for i, scene_id in enumerate(scenario_ids):
            if scene_id not in self.challenge_submission.predictions:
                self.challenge_submission.predictions[scene_id] = {}
            
            for j in range(num_agents):
                track_id = track_ids[i, j] if hasattr(track_ids[i], '__getitem__') else track_ids[i]
                self.challenge_submission.predictions[scene_id][track_id] = (
                    global_trajectory[i, j], 
                    probability[i, j]
                )
    # def format_data(
    # self,
    # data: dict,
    # trajectory: Tensor,
    # probability: Tensor,
    # normalized_probability=False,
    # inference=False,
    # ) -> None:
    #     """
    #     trajectory: (B, A, M, 60, 2) - Batch, Agents, Modes, Timesteps, Coordinates
    #     probability: (B, A, M) - Batch, Agents, Modes
    #     normalized_probability: if the input probability is normalized,
    #     """
    #     scenario_ids = data["scenario_id"]
    #     track_ids = data["track_id"]
    #     batch = trajectory.shape[0]
    #     num_agents = trajectory.shape[1]

    #     origin = data["origin"].view(batch, 1, 1, 2).double()
    #     print(f"Origin: {origin[0, :]}")  # Debugging line
    #     theta = data["theta"].double()
    #     print(f"Theta: {theta[0]}")  # Debugging line

    #     rotate_mat = torch.stack(
    #         [
    #             torch.cos(theta),
    #             torch.sin(theta),
    #             -torch.sin(theta),
    #             torch.cos(theta),
    #         ],
    #         dim=1,
    #     ).reshape(batch, 2, 2)

    #     print(f"Rotate matrix: {rotate_mat[0, :, :]}")  # Debugging line
    #     print(f"Trajectory shape: {trajectory.shape}")
    #     print(f"Trajectory: {trajectory[0, 0, :, :2].double()}")  # Debugging line
    #     print(f"Trajectory real shape: {data['y'].shape}")
    #     print(f"Trajectory real: {data['y'][0, 0, :, :2].double()}")  # Debugging line
    #     print(f"Data keys: {data.keys()}")  # Debugging line
    #     with torch.no_grad():
    #         # Transformación a coordenadas globales
    #         global_trajectory = (
    #             torch.matmul(trajectory[..., :2].double(), rotate_mat.unsqueeze(2))  # (B, A, M, 60, 2)
    #             + origin
    #         )
            
    #         if not normalized_probability:
    #             probability = torch.softmax(probability.double(), dim=-1)

    #     global_trajectory = global_trajectory.detach().cpu().numpy()
    #     probability = probability.detach().cpu().numpy()

    #     if inference:
    #         return global_trajectory, probability

    #     # Guardar predicciones para cada agente en cada escenario
    #     for i, scene_id in enumerate(scenario_ids):
    #         if scene_id not in self.challenge_submission.predictions:
    #             self.challenge_submission.predictions[scene_id] = {}
            
    #         for j in range(num_agents):
    #             track_id = track_ids[i, j] if isinstance(track_ids[i], (list, tuple, torch.Tensor)) else track_ids[i]
    #             self.challenge_submission.predictions[scene_id][track_id] = (
    #                 global_trajectory[i, j], 
    #                 probability[i, j]
    #             )

    # def format_data(
    #     self,
    #     data: dict,
    #     trajectory: Tensor,
    #     probability: Tensor,
    #     normalized_probability=False,
    #     inference=False,
    # ) -> None:
    #     """
    #     trajectory: (B, M, 60, 2)
    #     probability: (B, M)
    #     normalized_probability: if the input probability is normalized,
    #     """
    #     scenario_ids = data["scenario_id"]
    #     track_ids = data["track_id"]
    #     batch = len(track_ids)

    #     origin = data["origin"].view(batch, 1, 1, 2).double()
    #     theta = data["theta"].double()

    #     rotate_mat = torch.stack(
    #         [
    #             torch.cos(theta),
    #             torch.sin(theta),
    #             -torch.sin(theta),
    #             torch.cos(theta),
    #         ],
    #         dim=1,
    #     ).reshape(batch, 2, 2)

    #     with torch.no_grad():
    #         global_trajectory = (
    #             torch.matmul(trajectory[..., :2].double(), rotate_mat.unsqueeze(1))
    #             + origin
    #         )
    #         if not normalized_probability:
    #             probability = torch.softmax(probability.double(), dim=-1)

    #     global_trajectory = global_trajectory.detach().cpu().numpy()
    #     probability = probability.detach().cpu().numpy()

    #     if inference:
    #         return global_trajectory, probability

    #     for i, (scene_id, track_id) in enumerate(zip(scenario_ids, track_ids)):
    #         self.challenge_submission.predictions[scene_id] = {
    #             track_id: (global_trajectory[i], probability[i])
    #         }

    def generate_submission_file(self):
        print("generating submission file for argoverse 2 motion forecasting challenge")
        self.challenge_submission.to_parquet(self.submission_file)
        print(f"file saved to {self.submission_file}")
