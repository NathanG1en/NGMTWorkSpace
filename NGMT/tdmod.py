
import pandas as pd
import numpy as np


class TurnDetector:
    def __init__(self, angle_threshold=80, max_time_threshold=12):
        """
        Initialize the TurnDetector with thresholds for angle and maximum time.

        Parameters:
            angle_threshold (float): The angle in degrees to detect a turn.
            max_time_threshold (float): The maximum duration (in seconds) for a valid turn.
        """
        self.angle_threshold = angle_threshold
        self.max_time_threshold = max_time_threshold
        self.previous_yaw_angle = None  # To store the previous yaw angle
        self.turn_start_time = None  # To track the start time of the turn
        self.detected_turns = []  # Store detected turns

    def quaternion_to_yaw(self, q):
        """
        Convert quaternion to yaw angle.

        Parameters:
            q (tuple): Quaternion as (S, X, Y, Z).

        Returns:
            float: Yaw angle in degrees.
        """
        s, x, y, z = q
        # Calculate yaw in radians
        yaw = np.arctan2(2 * (y * s + x * z), 1 - 2 * (y ** 2 + x ** 2))
        # Convert to degrees
        yaw_degrees = np.degrees(yaw)
        return yaw_degrees

    def process_data(self, imu_data):
        """
        Process IMU data to detect turns based on yaw angle extracted from quaternion orientation.

        Parameters:
            imu_data (DataFrame): DataFrame containing 'Orientation_S', 'Orientation_X', 'Orientation_Y', 'Orientation_Z' and 'Time'.
        """
        if not isinstance(imu_data, pd.DataFrame) or \
                'Orientation_S' not in imu_data or \
                'Orientation_X' not in imu_data or \
                'Orientation_Y' not in imu_data or \
                'Orientation_Z' not in imu_data:
            raise ValueError(
                "Input data must be a DataFrame with 'Orientation_S', 'Orientation_X', 'Orientation_Y', 'Orientation_Z' columns.")

        self.previous_yaw_angle = None  # Reset for each new batch of IMU data
        self.turn_start_time = None  # Reset turn start time

        # Loop through the DataFrame to extract yaw from quaternion orientation data
        for i in range(len(imu_data)):
            # Extract quaternion data
            quaternion_data = imu_data[['Orientation_S', 'Orientation_X', 'Orientation_Y', 'Orientation_Z']].iloc[i]
            quaternion = tuple(quaternion_data)  # Create a tuple from the quaternion data

            # Calculate the yaw angle from the quaternion
            current_yaw_angle = self.quaternion_to_yaw(quaternion)

            # Check if this is the first iteration
            if self.previous_yaw_angle is None:
                self.previous_yaw_angle = current_yaw_angle
                continue

            # Calculate the change in yaw angle
            delta_yaw = current_yaw_angle - self.previous_yaw_angle

            # Normalize delta_yaw to handle wrap-around (if necessary)
            delta_yaw = (delta_yaw + 180) % 360 - 180  # Normalize to range [-180, 180]

            # Check if the absolute value of the change in yaw exceeds the threshold (start of a potential turn)
            if abs(delta_yaw) >= self.angle_threshold and self.turn_start_time is None:
                # Record the start time of the turn
                self.turn_start_time = imu_data['Time'].iloc[i]

            # If a turn is in progress, check its duration
            if self.turn_start_time is not None:
                elapsed_time = imu_data['Time'].iloc[i] - self.turn_start_time

                # Check if the turn completes in less than or equal to 12 seconds
                if elapsed_time <= self.max_time_threshold:
                    self.detected_turns.append((i, current_yaw_angle, elapsed_time))
                    print(
                        f"Turn detected at index {i}, yaw angle: {current_yaw_angle:.2f} degrees, time: {elapsed_time:.2f} seconds")
                    # Reset after detecting a valid turn
                    self.reset()
                elif elapsed_time > self.max_time_threshold:
                    # If the turn takes too long, reset and ignore
                    print(f"Turn ignored at index {i}, duration exceeded {self.max_time_threshold} seconds.")
                    self.reset()

            # Update the previous yaw angle for the next iteration
            self.previous_yaw_angle = current_yaw_angle

    def reset(self):
        """Reset the yaw angle and turn start time."""
        self.previous_yaw_angle = None
        self.turn_start_time = None

    def get_detected_turns(self):
        """Return the list of detected turns."""
        return self.detected_turns
