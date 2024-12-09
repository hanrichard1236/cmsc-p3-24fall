import numpy as np
from filterpy.kalman import KalmanFilter as kalmanfilter


CAR_LENGTH = 40
CAR_WIDTH = 20
class KalmanFilter:
    def __init__(self, car, variance, width):
        """
        Constructs a kalman filter
        variance: variance of gaussian GPS noise
        width: width of uniform GPS noise
        You need to implement state transition matrix, measure ment matrix, measurement noise covariance, 
        process noise covariance, initial state estimate, and initial covariance matrix
        """
        self.kf = kalmanfilter(dim_x=4, dim_z=2)  # 4 states (x, y, velocity_x, velocity_y), 2 measurements (x, y)
        self.variance = variance
        self.width = width
        self.car = car

        # BEGIN_YOUR_CODE ######################################################
        #state transition matrix
        self.kf.F = np.array([[1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        #measurement matrix
        self.kf.H = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]])
        #measurement noise covariance matrix
        self.kf.R = np.array([[variance, 0],
                               [0, variance]])
        #process noise covariance
        self.kf.Q = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]) * width
        #intial state matrix
        self.kf.x = np.array([car.pos[0], car.pos[1], car.vel[0], car.vel[1]])
        #initial covariance matrix
        self.kf.P = np.eye(4) * 1000

        # END_YOUR_CODE ########################################################

    def predict_and_update(self, measurement, other_car, which="gaussian"):
        """
        Returns the state after predicting and updating
        measurement: GPS measurement
        which: gaussian or uniform
        other_car_pos: position of the other car, used for check collision and update velocity
        """
        # Prediction step
        
        # BEGIN_YOUR_CODE ######################################################
        # Collision check to update velocity
        self.check_collision(other_car)

        # Prediction step
        self.kf.predict()

        # Update step
        if which == "gaussian":
            self.kf.update(measurement)
        else:
            self.kf.R = np.array([[2.0, 0],
                               [0, 2.0]])
            self.kf.update(measurement)
        
        # END_YOUR_CODE ########################################################

        return self.kf.x
    
    def check_collision(self, car):
        car1_corners = self.get_car_corners(self.car)
        car2_corners = self.get_car_corners(car)

        if self.rectangles_collide(car1_corners, car2_corners):
            collision_normal = self.car.pos - car.pos
            
            norm = np.linalg.norm(collision_normal)
            if norm > 1e-10:
                collision_normal /= norm
            else:
                collision_normal = np.array([1.0, 0.0])

            relative_velocity = car.vel - self.car.vel
            impulse = 2 * np.dot(relative_velocity, collision_normal) * collision_normal
            self.car.vel += impulse / 2
            car.vel -= impulse / 2

            self.car.vel *= 0.8
            car.vel *= 0.8

    def get_car_corners(self, car):
        half_length = CAR_LENGTH / 2
        half_width = CAR_WIDTH / 2
        cos_theta = car.orient[0]
        sin_theta = car.orient[1]

        corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])

        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        rotated_corners = np.dot(corners, rotation_matrix.T)
        return rotated_corners + car.pos

    def rectangles_collide(self, corners1, corners2):
        for shape in [corners1, corners2]:
            for i in range(len(shape)):
                axis = np.array([-shape[i-1][1] + shape[i][1], shape[i-1][0] - shape[i][0]])
                axis /= np.linalg.norm(axis)

                min1, max1 = float('inf'), float('-inf')
                min2, max2 = float('inf'), float('-inf')

                for corner in corners1:
                    projection = np.dot(corner, axis)
                    min1 = min(min1, projection)
                    max1 = max(max1, projection)

                for corner in corners2:
                    projection = np.dot(corner, axis)
                    min2 = min(min2, projection)
                    max2 = max(max2, projection)

                if max1 < min2 or max2 < min1:
                    return False

        return True