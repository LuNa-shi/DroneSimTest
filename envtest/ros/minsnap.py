import numpy as np
from typing import List, Dict, Optional
import rowan

class Point:
    def __init__(self, position: np.ndarray, velocity: np.ndarray = None, 
                 acceleration: np.ndarray = None, jerk: np.ndarray = None, 
                 time_from_start: float = 0.0):
        self.position = position if position is not None else np.zeros(3)
        self.velocity = velocity if velocity is not None else np.zeros(3)
        self.acceleration = acceleration if acceleration is not None else np.zeros(3)
        self.jerk = jerk if jerk is not None else np.zeros(3)
        self.time_from_start = time_from_start

class TrajectoryExt:
    def __init__(self):
        self.frame_id = None
        self.points = []
        self.poly_order = None
        self.continuity_order = None
        self.poly_coeff = []

    def set_frame(self, frame_id):
        """Set the frame ID for the trajectory."""
        self.frame_id = frame_id

    def fit_polynomial_coeffs(self, poly_order: int, continuity_order: int):
        # Reset polynomial coefficients
        self.poly_order = poly_order
        self.continuity_order = continuity_order
        self.poly_coeff = []

        # Check if first point starts from zero time
        if self.points[0].time_from_start != 0.0:
            print("Cannot fit polynomial when first point does not start from zero!")
            return

        # Scaling factors
        scaling = [1.0, 1.0, 0.5, 1.0 / 6.0]

        # Initialize polynomial coefficients with zeros
        for _ in range(poly_order + 1):
            self.poly_coeff.append(np.zeros(3))

        # Set constraints at the beginning
        for axis in range(3):
            for cont_idx in range(self.continuity_order + 1):
                if cont_idx == 0:
                    self.poly_coeff[cont_idx][axis] = scaling[cont_idx] * self.points[0].position[axis]
                elif cont_idx == 1:
                    # Velocity
                    self.poly_coeff[cont_idx][axis] = scaling[cont_idx] * self.points[0].velocity[axis]
                    if np.linalg.norm(self.points[0].velocity) < 0.1 and axis == 0:
                        self.poly_coeff[cont_idx][axis] += 0.2
                elif cont_idx == 2:
                    # Acceleration
                    self.poly_coeff[cont_idx][axis] = scaling[cont_idx] * self.points[0].acceleration[axis]   
                elif cont_idx == 3:
                    # Jerk
                    self.poly_coeff[cont_idx][axis] = scaling[cont_idx] * self.points[0].jerk[axis]

        # Check trajectory length
        if len(self.points) < 3:
            print(f"Trajectory of length [{len(self.points)}] too short to compute meaningful fit!")
            return

        # Prepare matrices for least squares solving
        A = np.zeros((len(self.points) - 1, poly_order - continuity_order))
        
        # Solve for each axis
        for axis in range(3):
            b = np.zeros(len(self.points) - 1)
            
            for i in range(1, len(self.points)):
                t = self.points[i].time_from_start
                
                # Fill matrix A
                for j in range(continuity_order + 1, poly_order + 1):
                    A[i-1, j - (continuity_order + 1)] = t ** j
                
                # Prepare right side of equation
                b[i-1] = self.points[i].position[axis]
                
                # Subtract known coefficients
                for cont_idx in range(continuity_order + 1):
                    b[i-1] -= self.poly_coeff[cont_idx][axis] * (t ** cont_idx)
            
            # Solve least squares
            x_coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Update polynomial coefficients
            for j in range(continuity_order + 1, poly_order + 1):
                self.poly_coeff[j][axis] = x_coeff[j - (continuity_order + 1)]

        self.poly_coeff = np.array(self.poly_coeff)
        return self.poly_coeff
    
    def evaluate_poly(self, dt, derivative):
        if self.poly_coeff is None:
            raise ValueError("Polynomial coefficients not computed")
        result = np.zeros(3)
        for axis in range(3):
            print(self.poly_coeff)
            print(axis)
            axis_coeffs = self.poly_coeff[:, axis]
            if derivative == 0:
                result[axis] = sum(coeff * (dt ** j) for j, coeff in enumerate(axis_coeffs))
            elif derivative == 1:
                result[axis] = sum(
                    j * coeff * (dt ** (j - 1)) 
                    for j, coeff in enumerate(axis_coeffs[1:], start=1))
            elif derivative == 2:
                result[axis] = sum(
                    j * (j - 1) * coeff * (dt ** (j - 2)) 
                    for j, coeff in enumerate(axis_coeffs[2:], start=2))   
            elif derivative == 3:
                result[axis] = sum(
                    j * (j - 1) * (j - 2) * coeff * (dt ** (j - 3)) 
                    for j, coeff in enumerate(axis_coeffs[3:], start=3))
        return result
    
    def resample_points_from_poly_coeffs(self):
        for point in self.points:
            # Evaluate polynomial derivatives
            point.position = self.evaluate_poly(point.time_from_start, 0)
            point.velocity = self.evaluate_poly(point.time_from_start, 1)
            point.acceleration = self.evaluate_poly(point.time_from_start, 2)
            point.jerk = self.evaluate_poly(point.time_from_start, 3)

            # if self.frame_id != FrameID.World:
            #     continue

            thrust = point.acceleration + 9.81 * np.array([0.0, 0.0, 1.0])
            I_eZ_I = np.array([0.0, 0.0, 1.0])
            
            q_pitch_roll = rowan.from_matrix(rowan.to_matrix(rowan.rotation_from_two_vectors(I_eZ_I, thrust)))
            
            # Compute linear velocity in body frame
            linvel_body = rowan.rotate(rowan.inverse(q_pitch_roll), point.velocity)
            
            # Compute heading
            heading = 0.0
            if self.yawing_enabled:
                heading = np.arctan2(point.velocity[1], point.velocity[0])
            
            # Compute attitude quaternion
            q_heading = rowan.from_axis_angle(np.array([0.0, 0.0, 1.0]), heading)
            q_att = rowan.multiply(q_pitch_roll, q_heading)
            point.attitude = rowan.normalize(q_att)

            # Compute thrust and body rates
            point.collective_thrust = np.linalg.norm(thrust)
            
            time_step = 0.1
            thrust_1 = thrust - time_step / 2.0 * point.jerk
            thrust_2 = thrust + time_step / 2.0 * point.jerk
            
            thrust_1 /= np.linalg.norm(thrust_1)
            thrust_2 /= np.linalg.norm(thrust_2)
            
            cross_prod = np.cross(thrust_1, thrust_2)
            angular_rates_wf = np.zeros(3)
            
            if np.linalg.norm(cross_prod) > 0.0:
                dot_product = np.dot(thrust_1, thrust_2)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                
                angular_rates_wf = (
                    np.arccos(dot_product) / time_step * 
                    cross_prod / (np.linalg.norm(cross_prod) + 1.0e-5)
                )
            
            point.bodyrates = rowan.rotate(rowan.inverse(point.attitude), angular_rates_wf)
    
    def set_constant_arc_length_speed(self, speed: float, traj_len: int, traj_dt: float) -> bool:
        # Check if polynomial coefficients are available
        if self.poly_coeff is None or len(self.poly_coeff) == 0:
            print("Polynomial coefficients unknown!")
            return False

        # Initialize new points with the first point
        new_points = [self.points[0]]
        start_time = 0.0
        end_time = self.points[-1].time_from_start
        steps = 100
        dt = (end_time - start_time) / steps

        # Resampling variables
        j = 1  # Point index
        t_curr = start_time + dt
        pos_prev = np.array([self.evaluate_poly(t_curr, axis) for axis in range(3)])
        acc_arc_length = 0.0

        while t_curr <= end_time:
            # Sample polynomial and compute arc length
            pos_curr = np.array([self.evaluate_poly(t_curr, axis) for axis in range(3)])
            arc_length_increment = np.linalg.norm(pos_curr - pos_prev)
            acc_arc_length += arc_length_increment

            # Add point when desired arc length is reached
            if acc_arc_length >= speed * j * traj_dt:
                new_point = Point(
                    position=pos_curr, 
                    time_from_start=j * traj_dt
                )
                new_points.append(new_point)
                j += 1

                # Stop if we have enough points
                if len(new_points) >= len(self.points):
                    break

            pos_prev = pos_curr
            t_curr += dt

        # Extrapolate if not enough points
        if j < 2:
            print("Not enough points to extrapolate, won't adapt trajectory!")
            return False

        while len(new_points) < len(self.points):
            last_point = new_points[-1]
            second_last_point = new_points[-2]
            
            # Extrapolate with constant velocity
            extrapolated_pos = (
                last_point.position + 
                (last_point.position - second_last_point.position)
            )
            
            new_point = Point(
                position=extrapolated_pos, 
                time_from_start=j * traj_dt
            )
            new_points.append(new_point)
            j += 1

        # Update points and refit polynomial
        self.points = new_points
        self.fit_polynomial_coeffs(self.poly_order, self.continuity_order)
        
        # Optional: Implement resample_points_from_poly_coeffs if needed
        # self.resample_points_from_poly_coeffs()

        return True