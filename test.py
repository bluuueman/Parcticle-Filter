import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats
import matplotlib.pyplot as plt

np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)
import cv2


def drawLines(img, points, r, g, b):
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))


def drawCross(img, center, r, g, b):
    d = 5
    t = 2
    LINE_AA = cv2.LINE_AA if cv2.__version__[0] == '4' else cv2.CV_AA
    color = (r, g, b)
    ctrx = center[0, 0]
    ctry = center[0, 1]
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)


def mouseCallback(event, x, y, flags, null):
    global center
    global trajectory
    global previous_x
    global previous_y
    global zs

    center = np.array([[x, y]])
    trajectory = np.vstack((trajectory, np.array([x, y])))
    # noise=sensorSigma * np.random.randn(1,2) + sensorMu

    if previous_x > 0:
        heading = np.arctan2(np.array([y - previous_y]), np.array([previous_x - x]))

        if heading > 0:
            heading = -(heading - np.pi)
        else:
            heading = -(np.pi + heading)

        distance = np.linalg.norm(np.array([[previous_x, previous_y]]) - np.array([[x, y]]), axis=1)

        std = np.array([2, 4])
        u = np.array([heading, distance])
        predict(particles, u, std, dt=1.)
        #robot和landmark之间的距离，其中 np.random.randn(NL) * sensor_std_err 为随机误差
        zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))
        update(particles, weights, z=zs, R=50, landmarks=landmarks)

        indexes = systematic_resample(weights)
        resample_from_index(particles, weights, indexes)

    previous_x = x
    previous_y = y


WIDTH = 800
HEIGHT = 600
WINDOW_NAME = "Particle Filter"

# sensorMu=0
# sensorSigma=3

sensor_std_err = 5

#生成粒子
def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles

#预测系统状态：更新粒子位置
def predict(particles, u, std, dt=1.):
    N = len(particles)
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    particles[:, 0] += np.cos(u[0]) * dist
    particles[:, 1] += np.sin(u[0]) * dist

#更新粒子权重
def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        distance = np.power((particles[:, 0] - landmark[0]) ** 2 + (particles[:, 1] - landmark[1]) ** 2, 0.5)
        '''
        权值通过正态分布进行修正，若使用帕累托分布代码改为以下
        weights *= scipy.stats.pareto(1).pdf(0.1*abs(z[i]-distance)+1，1)
        说明：因为帕累托分布x大于等于1，因此令x=0.1*abs(z[i]-distance)+1,差值乘以0.1因为该分布较为集中在[1,2]之间
        实际移动范围却远大于1，乘以0.1保证了粒子多样性，对位置的估计更准确
        '''
        weights *= scipy.stats.norm(distance,R).pdf(z[i])

    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)

#判断粒子有效性，可以在粒子有效性低于一定阈值时才进行重采样，可以保证粒子多样性
def neff(weights):
    return 1. / np.sum(np.square(weights))

#对粒子根据与robot的相对距离进行重采样
def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)

#通过粒子位置的加权平均数估计robot位置
def estimate(particles, weights):
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean


x_range = np.array([0, 800])
y_range = np.array([0, 600])

# Number of partciles
N = 400

landmarks = np.array([[144, 73], [410, 13], [336, 175], [718, 159], [178, 484], [665, 464]])
NL = len(landmarks)
particles = create_uniform_particles(x_range, y_range, N)

weights = np.array([1.0] * N)

# Create a black image, a window and bind the function to window
img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouseCallback)

center = np.array([[-10, -10]])

trajectory = np.zeros(shape=(0, 2))
robot_pos = np.zeros(shape=(0, 2))
previous_x = -1
previous_y = -1
DELAY_MSEC = 50

while (1):

    cv2.imshow(WINDOW_NAME, img)
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    drawLines(img, trajectory, 0, 255, 0)
    drawCross(img, center, r=255, g=0, b=0)

    # landmarks
    for landmark in landmarks:
        cv2.circle(img, tuple(landmark), 10, (255, 0, 0), -1)

    # draw_particles:
    for particle in particles:
        cv2.circle(img, tuple((int(particle[0]), int(particle[1]))), 1, (255, 255, 255), -1)

    if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
        break

    estimate_position=estimate(particles,weights)
    cv2.circle(img, ((int(estimate_position[0])),(int(estimate_position[1]))), 4,(0,0,255), -1)#显示estimate position具体位置

    cv2.circle(img, (10, 10), 10, (255, 0, 0), -1)
    cv2.circle(img, (10, 30), 3, (255, 255, 255), -1)
    cv2.circle(img, (10, 75), 4, (0, 0, 255), -1)#显示estimate position图标
    cv2.putText(img, "Landmarks", (30, 20), 1, 1.0, (255, 0, 0))
    cv2.putText(img, "Particles", (30, 40), 1, 1.0, (255, 255, 255))
    cv2.putText(img, "Robot Trajectory(Ground truth)", (30, 60), 1, 1.0, (0, 255, 0))
    cv2.putText(img, "Robot Estimated Position", (30, 80), 1, 1.0, (0, 0, 255))#显示estimate position的文字
    drawLines(img, np.array([[10, 55], [25, 55]]), 0, 255, 0)


cv2.destroyAllWindows()