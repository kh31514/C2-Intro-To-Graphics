import numpy as np
import math
from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in 
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might 
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None, texture=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
          texture : array representing the material's texture
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d
        self.texture = texture


class Hit:

    def __init__(self, t, point=None, normal=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material


# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """

        # calculate coefficients for quadratic equation representing intersection

        c = np.dot(ray.origin-self.center, ray.origin -
                   self.center) - self.radius**2
        b = 2*np.dot(ray.origin-self.center, ray.direction)
        a = np.dot(ray.direction, ray.direction)

        # solve quadratic formula for t
        if b**2-4*a*c < 0:
            return no_hit
        else:
            t_1 = (-1*b + np.sqrt(b**2-4*a*c))/(2*a)
            t_2 = (-1*b - np.sqrt(b**2-4*a*c))/(2*a)
            t_vals = []
            if t_1 >= ray.start and t_1 <= ray.end:
                t_vals.append(t_1)
            if t_2 >= ray.start and t_2 <= ray.end:
                t_vals.append(t_2)
            if len(t_vals) == 0:
                return no_hit
            t = np.min(t_vals)
            point = []
            for i in range(3):
                point += [ray.origin[i]+t*ray.direction[i]]
            normal = np.array(point) - self.center
            normal = normal/np.linalg.norm(normal)
            return Hit(t, point, normal, self.material)

class Cone:

    def __init__(self, center, r, height, material):
        """Create a cone with the given center (x_0, y_0, z_0) and value for c

        equation used: ((x-x_0)^2 + (z-z_0)^2)/r^2 = (y - y_0)^2

        Thus, the cone must point in the y direction

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          r : float -- a Python float specifying the value of r in the equation (affects cone radius)
          heigh : float -- a Python float restricting the height of the cone
          material : Material -- the material of the surface
        """
        self.center = center
        self.r = r
        self.height = height
        self.material = material

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this cone.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """

        # calculate coefficients for quadratic equation representing intersection

        a = ray.direction[0]**2 + ray.direction[2]**2 - self.r**2 * ray.direction[1]**2
        b = 2*ray.direction[0]*(ray.origin[0]-self.center[0]) + 2*ray.direction[2]*(ray.origin[2]-self.center[2]) 
        - self.r**2 * 2 * ray.direction[1] * (ray.origin[1]-self.center[1])
        c = (ray.origin[0]-self.center[0])**2 + (ray.origin[2]-self.center[2])**2 - self.r**2*(ray.origin[1]-self.center[1])**2

        # solve quadratic formula for t
        if b**2-4*a*c < 0:
            return no_hit
        else:
            t_1 = (-1*b + np.sqrt(b**2-4*a*c))/(2*a)
            t_2 = (-1*b - np.sqrt(b**2-4*a*c))/(2*a)
            t_vals = []
            if t_1 >= ray.start and t_1 <= ray.end:
                t_vals.append(t_1)
            if t_2 >= ray.start and t_2 <= ray.end:
                t_vals.append(t_2)
            if len(t_vals) == 0:
                return no_hit
            t = np.min(t_vals)
            point = []
            for i in range(3):
                point += [ray.origin[i]+t*ray.direction[i]]
            # make sure the point is within the y bounds for the cone
            #if point[1] < self.center[1] or point[1] > self.center[1] + self.height:
                #return no_hit
            # calculate surface normal
            vec_a = point - self.center
            vec_b = point - np.array([self.center[0], point[1], self.center[2]])
            tangent = np.cross(vec_a, vec_b)
            normal = normalize(np.cross(tangent, vec_a))
            if normal[1] > 0:
                # make sure normal points in the right direction
                normal *= -1
            return Hit(t, point, normal, self.material)

class Cylinder:

  def __init__(self, center, radius, height, theta, material):
        """Create a cylinder with the given center, radius, and height rotated theta degrees with respect to the z axis.

        Parameters:
          center : (3,) -- a 3D point specifying the cylinder's center
          radius : float -- a Python float specifying the cylinder's radius
          height : float -- a Python float specifying the cylinder's height
          theta : float -- a Python float specifying the cylinder's degree of rotation with respect to the z axis (measured in degrees, not radians)
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.height = height
        self.theta = theta * np.pi / 180
        self.material = material

  def rotate_point_around_z(self, point, angle):
    x = point[0] * math.cos(angle) - point[1] * math.sin(angle)
    y = point[0] * math.sin(angle) + point[1] * math.cos(angle)
    return np.array([x, y, point[2]])
  
  def inverse_rotate_point_around_z(self, point, angle):
    x = point[0] * math.cos(angle) + point[1] * math.sin(angle)
    y = -point[0] * math.sin(angle) + point[1] * math.cos(angle)
    return np.array([x, y, point[2]])
  
  def intersect(self, ray:Ray):
    """Computes the intersection between a ray and this cylinder, if it exists.

      Parameters:
        ray : Ray -- the ray to intersect with the triangle
      Return:
        Hit -- the hit data
      """
    # Ray parameters
    O = self.rotate_point_around_z(ray.origin, -self.theta)
    D = self.rotate_point_around_z(ray.direction, -self.theta)

    # Cylinder parameters
    C = self.rotate_point_around_z(self.center, -self.theta)
    r = self.radius
    h = self.height

    # Calculate components for solving quadratic equation
    a = D[0] * D[0] + D[2] * D[2]  # Assuming a cylinder aligned with y-axis
    b = 2 * (D[0] * (O[0] - C[0]) + D[2] * (O[2] - C[2]))
    c = (O[0] - C[0]) * (O[0] - C[0]) + (O[2] - C[2]) * (O[2] - C[2]) - r * r

    # Solve the quadratic equation
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return no_hit  # No intersection

    # Calculate the closest intersection point
    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)

    # Check if the intersection points are within bounds of the cylinder
    intersection1 = O + t1 * D
    intersection2 = O + t2 * D

    # Check if intersections are within height bounds
    y_min = C[1]
    y_max = C[1] + h
    if y_min > y_max:
        y_min, y_max = y_max, y_min  # Swap values if y_min is greater

    valid_intersection1 = y_min <= intersection1[1] <= y_max
    valid_intersection2 = y_min <= intersection2[1] <= y_max

    if valid_intersection1 and valid_intersection2:
        if (np.linalg.norm(intersection1 - O))**2 < (np.linalg.norm(intersection2 - O))**2:
          normal_x = 2 * (intersection1[0] - C[0])
          normal_y = 0  # Cylinder is aligned with the y-axis, so no change in y direction
          normal_z = 2 * (intersection1[2] - C[2])
          normal_vector = (normal_x, normal_y, normal_z)
          normal_vector = self.inverse_rotate_point_around_z(normal_vector, self.theta)
          normal_vector = normalize(normal_vector)
          intersection1 = self.inverse_rotate_point_around_z(intersection1, self.theta)
          return Hit(t1, point=intersection1, normal=normal_vector, material=self.material)
        else:
          normal_x = 2 * (intersection2[0] - C[0])
          normal_y = 0  # Cylinder is aligned with the y-axis, so no change in y direction
          normal_z = 2 * (intersection2[2] - C[2])
          normal_vector = (normal_x, normal_y, normal_z)
          normal_vector = self.inverse_rotate_point_around_z(normal_vector, self.theta)
          normal_vector = normalize(normal_vector)
          intersection2 = self.inverse_rotate_point_around_z(intersection2, self.theta)
          return Hit(t2, point=intersection2, normal=normal_vector, material=self.material)
    elif valid_intersection1:
        normal_x = 2 * (intersection1[0] - C[0])
        normal_y = 0  # Cylinder is aligned with the y-axis, so no change in y direction
        normal_z = 2 * (intersection1[2] - C[2])
        normal_vector = (normal_x, normal_y, normal_z)
        normal_vector = self.inverse_rotate_point_around_z(normal_vector, self.theta)
        normal_vector = normalize(normal_vector)
        intersection1 = self.inverse_rotate_point_around_z(intersection1, self.theta)
        return Hit(t1, point=intersection1, normal=normal_vector, material=self.material)
    elif valid_intersection2:
        normal_x = 2 * (intersection2[0] - C[0])
        normal_y = 0  # Cylinder is aligned with the y-axis, so no change in y direction
        normal_z = 2 * (intersection2[2] - C[2])
        normal_vector = (normal_x, normal_y, normal_z)
        normal_vector = self.inverse_rotate_point_around_z(normal_vector, self.theta)
        normal_vector = normalize(normal_vector)
        intersection2 = self.inverse_rotate_point_around_z(intersection2, self.theta)
        return Hit(t2, point=intersection2, normal=normal_vector, material=self.material)
    else:
        return no_hit  # No intersection within cylinder bounds

class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        m = np.array([[self.vs[0][0]-self.vs[1][0], self.vs[0][0]-self.vs[2][0], ray.direction[0]],
                      [self.vs[0][1]-self.vs[1][1], self.vs[0]
                          [1]-self.vs[2][1], ray.direction[1]],
                      [self.vs[0][2]-self.vs[1][2], self.vs[0][2]-self.vs[2][2], ray.direction[2]]])
        result_matrix = np.array(
            [self.vs[0][0]-ray.origin[0], self.vs[0][1]-ray.origin[1], self.vs[0][2]-ray.origin[2]])
        solutions = np.linalg.solve(m, result_matrix)
        beta = solutions[0]
        gamma = solutions[1]
        t = solutions[2]

        if t >= ray.start and t <= ray.end and beta > 0 and gamma > 0 and beta+gamma < 1:
            normal = np.cross(self.vs[0]-self.vs[1], self.vs[0]-self.vs[2])
            return Hit(t, ray.origin + t*ray.direction, normal, self.material)
        return no_hit


class Camera:

    def __init__(self, eye=vec([0, 0, 0]), target=vec([0, 0, -1]), up=vec([0, 1, 0]),
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        self.f = None  # you should set this to the distance from your center of projection to the image plane
        # set this to the matrix that transforms your camera's coordinate system to world coordinates
        self.M = np.eye(4)
        self.target = target
        self.up = up
        self.vfov = vfov

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
                      (note: since we initially released this code with specs saying 0,0 was at the bottom left, we will
                       accept either convention for this assignment)
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        # TODO A4 implement this function
        # generate ray using perspective

        d = np.linalg.norm(self.eye-self.target)
        h = 2*d*np.tan(self.vfov/2.*np.pi/180.)
        w = h*self.aspect

        text_coords = np.append(img_point, 1)
        # scale and translate to origin
        m = np.array([[w, 0, -1.*w/2,], [0, -1.*h, h/2], [0, 0, 1.]])
        img_coords = m @ text_coords
        u = img_coords[0]
        v = img_coords[1]
        
        w_vec = normalize(self.eye-self.target)
        u_vec = normalize(np.cross(self.up, w_vec))
        v_vec = normalize(np.cross(w_vec, u_vec))
        

        # subtract camera location from image point
        ray_dir = -1*d*w_vec + u * u_vec + v * v_vec

        return Ray(self.eye, ray_dir, 0)


class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray: Ray, hit: Hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """

        #return np.zeros(3)

        l = self.position - hit.point
        r = np.linalg.norm(l)
        l /= np.linalg.norm(l)
        n = hit.normal / np.linalg.norm(hit.normal)
        v = -ray.direction / np.linalg.norm(ray.direction)
        h = (v + l) / np.linalg.norm(v + l)

        blocking = scene.intersect(
            Ray(origin=hit.point, direction=self.position-hit.point, start=10**-6))
        if blocking != no_hit:
            return np.zeros(3)

        for surf in scene.surfs:
            point = np.array(surf.intersect(ray).point)
            if (point == hit.point).all():
                diffuse = surf.material.k_d
                if type(surf) == Sphere and hit.material.texture is not None:
                    # calculate texture value for sphere
                    # calculate spherical coordinates
                    point_vec = point - surf.center
                    x_vec = np.array([surf.radius, 0, 0])
                    y_vec = np.array([0, surf.radius, 0])
                    point_vec_proj = np.array([point_vec[0], 0, point_vec[2]])
                    dot = np.dot(x_vec, point_vec_proj)
                    phi = np.arccos(np.dot(x_vec, point_vec_proj)/(np.linalg.norm(x_vec)*np.linalg.norm(point_vec_proj)))
                    theta = np.arccos(np.dot(y_vec, point_vec)/(np.linalg.norm(y_vec)*np.linalg.norm(point_vec)))

                    
                    y = theta/np.pi*hit.material.texture.shape[0]
                    x = phi/np.pi*hit.material.texture.shape[1]
                    y = int(np.round(y))
                    x = int(np.round(x))

                    # check for out of bounds
                    if x == hit.material.texture.shape[1]:
                        x -= 1
                    if y == hit.material.texture.shape[0]:
                        y -= 1

                    diffuse = hit.material.texture[y,x]
                    diffuse = diffuse/255

                shading = (diffuse + surf.material.k_s * (n @ h)**surf.material.p) * self.intensity * \
                    np.clip((n @ l),
                            0, None) / r**2
        return shading


class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function

        return hit.material.k_a * self.intensity


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2, 0.3, 0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function

        intersections = []
        for surf in self.surfs:
            intersections.append(surf.intersect(ray))

        smallest_t_intersection = no_hit
        for intersection in intersections:
            if intersection.t < smallest_t_intersection.t:
                smallest_t_intersection = intersection

        return smallest_t_intersection


MAX_DEPTH = 4


def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    # TODO A4 implement this function
    if hit == no_hit:  # indicates no hit
        return scene.bg_color

    output = vec([0, 0, 0])

    if depth == MAX_DEPTH:
        return output

    # mirror reflection
    if hit.material.k_m > 0:
        v = ray.direction
        r = 2 * np.dot(hit.normal, v) * hit.normal - v
        refl_ray = Ray(hit.point, -1*r, 10**-6)
        reflection = shade(refl_ray, scene.intersect(
            refl_ray), scene, lights, depth+1)
        output += hit.material.k_m*reflection

    for light in lights:
        output += light.illuminate(ray, hit, scene)
    return output


def render_image(camera: Camera, scene: Scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    # TODO A4 implement this function

    output = np.zeros((ny, nx, 3), np.float32)

    for i in range(ny):
        for j in range(nx):
            # calculate world coordinates
            texture_coords = np.array([(j+.5)/nx, (i+.5)/ny])
            ray = camera.generate_ray(texture_coords)

            hit = scene.intersect(ray)
            output[i][j] = shade(ray, hit, scene, lights)

    return output
