import numpy as np

class Building:
    def __init__(self, id, location, length, width, height):
        self.id = id
        self.location = location
        self.length = length
        self.width = width
        self.height = height
        self.planes = []


        # Z-plane
        self.planes.append({'eqn': [0, 0, 1, self.location[2] + self.height], 'points': [
                            [self.location[0] + self.length/2, self.location[1] + self.width/2, self.location[2] + self.height],
                            [self.location[0] - self.length/2, self.location[1] + self.width/2, self.location[2] + self.height],
                            [self.location[0] + self.length/2, self.location[1] - self.width/2, self.location[2] + self.height],
                            [self.location[0] - self.length/2, self.location[1] - self.width/2, self.location[2] + self.height]]})

        self.planes.append({'eqn': [0, 0, 1, self.location[2]], 'points': [
                            [self.location[0] + self.length/2, self.location[1] + self.width/2, self.location[2]],
                            [self.location[0] - self.length/2, self.location[1] + self.width/2, self.location[2]],
                            [self.location[0] + self.length/2, self.location[1] - self.width/2, self.location[2]],
                            [self.location[0] - self.length/2, self.location[1] - self.width/2, self.location[2]]]})

        # X-plane
        self.planes.append({'eqn': [1, 0, 0, self.location[0] + self.length/2], 'points': [
                            [self.location[0] + self.length/2, self.location[1] + self.width/2, self.location[2] + self.height],
                            [self.location[0] + self.length/2, self.location[1] + self.width/2, self.location[2]],
                            [self.location[0] + self.length/2, self.location[1] - self.width/2, self.location[2] + self.height],
                            [self.location[0] + self.length/2, self.location[1] - self.width/2, self.location[2]]]})


        self.planes.append({'eqn': [1, 0, 0, self.location[0] - self.length/2], 'points': [
                            [self.location[0] - self.length/2, self.location[1] + self.width/2, self.location[2] + self.height],
                            [self.location[0] - self.length/2, self.location[1] + self.width/2, self.location[2]],
                            [self.location[0] - self.length/2, self.location[1] - self.width/2, self.location[2] + self.height],
                            [self.location[0] - self.length/2, self.location[1] - self.width/2, self.location[2]]]})


        # Y-plane
        self.planes.append({'eqn': [0, 1, 0, self.location[1] + self.width/2], 'points': [
                            [self.location[0] + self.length/2, self.location[1] + self.width/2, self.location[2] + self.height],
                            [self.location[0] - self.length/2, self.location[1] + self.width/2, self.location[2] + self.height],
                            [self.location[0] + self.length/2, self.location[1] + self.width/2, self.location[2]],
                            [self.location[0] - self.length/2, self.location[1] + self.width/2, self.location[2]]]})


        self.planes.append({'eqn': [0, 1, 0, self.location[1] - self.width/2], 'points': [
                            [self.location[0] + self.length/2, self.location[1] - self.width/2, self.location[2] + self.height],
                            [self.location[0] - self.length/2, self.location[1] - self.width/2, self.location[2] + self.height],
                            [self.location[0] + self.length/2, self.location[1] - self.width/2, self.location[2]],
                            [self.location[0] - self.length/2, self.location[1] - self.width/2, self.location[2]]]})

        
    def reset(self):
        pass
    
    def line_eqn(self, p1, p2):
        m = (p2 - p1)/(np.linalg.norm(p2 - p1))
        return [p1, m]

    def find_intersection(self, line, plane):
        [A, B, C, D] = plane['eqn']
        [p1, m] = line
        if(A*m[0] + B*m[1] + C*m[2] == 0):
            return None
        t = (D - (A*p1[0] + B*p1[1] + C*p1[2]))/(A*m[0] + B*m[1] + C*m[2])
        intersection_point = np.array([p1[0] + t*m[0], p1[1] + t*m[1], p1[2] + t*m[2]])
        return intersection_point

    def is_in_line(self, p1, p2, target_p):
        if((p1[0] <= target_p[0] <= p2[0]) or (p2[0] <= target_p[0] <= p1[0])):
            if((p1[1] <= target_p[1] <= p2[1]) or (p2[1] <= target_p[1] <= p1[1])):
                if((p1[2] <= target_p[2] <= p2[2]) or (p2[2] <= target_p[2] <= p1[2])):
                    return True
        return False

    def is_in_plane(self, plane, target_p):
        plane_p = plane['points']
        # x plane
        if(target_p[0] == plane_p[0][0] and target_p[0] == plane_p[1][0]):
            if((plane_p[2][1] <= target_p[1] <= plane_p[0][1]) or (plane_p[0][1] <= target_p[1] <= plane_p[2][1])):
                if((plane_p[1][2] <= target_p[2] <= plane_p[0][2]) or (plane_p[0][2] <= target_p[2] <= plane_p[1][2])):
                    return True

        # y-plane
        if(target_p[1] == plane_p[0][1] and target_p[1] == plane_p[1][1]):
            if((plane_p[1][0] <= target_p[0] <= plane_p[0][0]) or (plane_p[0][0] <= target_p[0] <= plane_p[1][0])):
                if((plane_p[2][2] <= target_p[2] <= plane_p[0][2]) or (plane_p[0][2] <= target_p[2] <= plane_p[2][2])):
                    return True
        # z-plane
        if(target_p[2] == plane_p[0][2] and target_p[2] == plane_p[1][2]):
            if((plane_p[1][0] <= target_p[0] <= plane_p[0][0]) or (plane_p[0][0] <= target_p[0] <= plane_p[1][0])):
                if((plane_p[2][1] <= target_p[1] <= plane_p[0][1]) or (plane_p[0][1] <= target_p[1] <= plane_p[2][1])):
                    return True
        return False

    def line_plane_intersect(self, p1, p2, plane):
        line = self.line_eqn(p1, p2)
        intersection_point = self.find_intersection(line, plane)
        if(np.any(intersection_point == None)):
            return False
        return (self.is_in_line(p1, p2, intersection_point) and self.is_in_plane(plane, intersection_point))

    def is_blocking(self, point_1, point_2):
        for i in range(6):
            if(self.line_plane_intersect(point_1, point_2, self.planes[i])):
                return True
        return False

    def plane_num(self, point_1, point_2):
        num_blocking_planes = 0
        for i in range(6):
            if(self.line_plane_intersect(point_1, point_2, self.planes[i])):
                num_blocking_planes += 1
        return num_blocking_planes

    def obstacle(self, point):
        if(self.location[0] - self.length/2 < point[0] < self.location[0] + self.length/2):
            if(self.location[1] - self.width/2 < point[1] < self.location[1] + self.width/2):
                if(self.location[2] < point[2] < self.location[2] + self.height):
                    return True
        return False
    
    def points(self):
        x = self.location[0]
        y = self.location[1]
        l = self.length
        w = self.width
        corner = (x - l / 2, y - w / 2)
        return [corner, l, w]