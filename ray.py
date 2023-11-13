import numpy as np

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

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d


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
            if t_1 > t_2:
                t = t_2
            else:
                t = t_1
            point = []
            for i in range(3):
                point += [ray.origin[i]+t*ray.direction[i]]
            normal = np.array(point) - self.center
            normal = normal/np.linalg.norm(normal)
            return Hit(t, point, normal, self.material)


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
        # TODO A4 implement this function
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

        img_point[1] = 1-img_point[1]
        d = np.linalg.norm(self.eye-self.target)
        h = d*np.tan(self.vfov/2)
        w = h*self.aspect

        text_coords = img_point
        # scale and translate to origin
        m = np.array([[w, 0, -1.*w/2,], [0, -1.*h, h/2], [0, 0, 1]])
        img_coords = m @ text_coords
        u = img_coords[0]
        v = img_coords[1]
        w_vec = (self.target-self.eye)/np.linalg.norm(self.target-self.eye)
        v_vec = self.up/np.linalg.norm(self.up)
        u_vec = np.cross(v_vec, w_vec)/np.linalg.norm(np.cross(v_vec, w_vec))

        # subtract camera location from image point
        ray_dir = d*w_vec + u * u_vec + v * v_vec
        ray_dir /= np.linalg.norm(ray_dir)

        # TODO find valid start location
        return Ray(self.eye, ray_dir, 1)


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
        # TODO A4 implement this function

        # if there is an intersection between hit point and light pos, there should be a shadow
        # TODO fix this - should be in a diff part of the code to change pixels on the "ground"/other
        # maybe put it outside this loop, or in a diff function
        # blocking = scene.intersect(Ray(self.position, light_direction))
        # if blocking != no_hit and blocking.point != hit.point:
        # return np.zeros(3)

        # return vec([0, 0, 0])

        l = self.position - hit.point
        r = np.linalg.norm(l)
        l /= np.linalg.norm(l)
        n = hit.normal / np.linalg.norm(hit.normal)
        v = ray.direction / np.linalg.norm(ray.direction)
        h = -v + l / np.linalg.norm(-v + l)

        for surf in scene.surfs:
            if surf.intersect(ray).point == hit.point:
                # Diffuse shading
                diffuse_shading = surf.material.k_d * \
                    self.intensity * \
                    np.clip((n @ l), 0, None) / r**2

                # Specular shading
                specular_shading = surf.material.k_d + surf.material.k_s * (n @ h)**surf.material.p * self.intensity * \
                    np.clip((n @ l),
                            0, None) / r**2

                return diffuse_shading + specular_shading


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

        for surf in scene.surfs:
            if surf.intersect(ray).point == hit.point:
                return surf.material.k_a * self.intensity


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
    if hit.point == None:  # indicates no hit
        return scene.bg_color

    output = vec([0, 0, 0])
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
            texture_coords = np.array([(j)/nx, (i)/ny, 1])
            ray = camera.generate_ray(texture_coords)

            # for surf in scene.surfs:
            hit = scene.intersect(ray)
            output[i][j] = shade(ray, hit, scene, lights)

    return output
