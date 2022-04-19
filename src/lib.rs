use rand::Rng;
use std::{ops, fmt};
use std::cmp;
// use image::Rgb;

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {x, y, z}
    }

    pub fn random() -> Self {
        Self {x: random_f64(),
              y: random_f64(),
              z: random_f64()}
    }

    pub fn random_rng(min: f64, max: f64) -> Self {
        Self {x: random_f64_rng(min, max),
              y: random_f64_rng(min, max),
              z: random_f64_rng(min, max)}
    }

    pub fn len(&self) -> f64 {
        (self.length_squared()).sqrt()
    }

    pub fn length_squared(&self) -> f64 {
        f64::powf(self.x, 2.0) + f64::powf(self.y, 2.0) + f64::powf(self.z, 2.0)
    }
}


impl ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self {
        Self {x: - self.x,
              y: - self.y,
              z: - self.z}
    }
}

impl ops::Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Self) -> Self {
        Self {x: self.x - other.x,
              y: self.y - other.y,
              z: self.z - other.z}
    }
}

impl ops::Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, other: Self) -> Self {
        Self {x: self.x + other.x,
              y: self.y + other.y,
              z: self.z + other.z}
    }
}

impl ops::AddAssign<Vec3> for Vec3 {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Self {
        Self {x: self.x * v.x,
              y: self.y * v.y,
              z: self.z * v.z}
    }
}

impl ops::Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, t: f64) -> Self {
        Self {x: self.x * t,
              y: self.y * t,
              z: self.z * t}
    }
}

impl ops::MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, t: f64) {
        *self = *self * t
    }
}

impl ops::Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, t: f64) -> Self{
        self * (1.0 / t)
    }
}

impl ops::DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, t: f64) {
        *self *= 1.0 / t;
    }
}

pub type Point3 = Vec3;     // 3D point3
pub type Color = Vec3;      // RGB color


// Vec3 Utility Functions

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}

impl Vec3 {
    pub fn dot(u: &Vec3, v: &Vec3) -> f64 {
        u.x * v.x +
        u.y * v.y +
        u.z * v.z
    }

    pub fn cross(u: &Vec3, v: &Vec3) -> Self {
        Self {x: u.y * v.z - u.z * v.y,
              y: u.z * v.x - u.x * v.z,
              z: u.x * v.y - u.y * v.x}
    }

    pub fn unit_vector(self) -> Vec3 {
        self / self.len()
    }

    pub fn reflect(v: &Vec3, n: &Vec3) -> Self {
        *v - *n * 2.0 * Vec3::dot(v, n)
    }

    pub fn near_zero(&self) -> bool {
        // Return true if the vector is close to zero in all dimensions
        let s = 1.0e-8_f64;
        (self.x.abs() < s) && (self.y.abs() < s) && (self.z.abs() < s)
    }
}

pub fn random_in_unit_sphere() -> Vec3 {
    loop {
        let p = Vec3::random_rng(-1.0, 1.0);
        if p.length_squared() >= 1.0 { continue };
        return p;
    }
}

pub fn random_unit_vector() -> Vec3 {
    let v = random_in_unit_sphere();
    v.unit_vector()
}

// pub fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
//     *v - *n * 2.0 * Vec3::dot(v, n)
// }

pub fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
    let dot_val = Vec3::dot(&(-*uv), n);
    let cos_theta = match dot_val.partial_cmp(&1.0).unwrap() {
        cmp::Ordering::Less => dot_val,
        _ => 1.0,
    };

    let r_out_perp: Vec3 = (*uv + *n * cos_theta) * etai_over_etat;
    let r_out_parallel = *n * (- ((1.0 - r_out_perp.length_squared()).sqrt()).abs());
    r_out_perp + r_out_parallel
}

// Color Utility Functions

impl Color {
    pub fn write_color(&self, samples_per_pixel: u32) -> (u8, u8, u8) {
        let mut r = self.x;
        let mut g = self.y;
        let mut b = self.z;

        // Divide the color by the nuber of samples
        let scale = 1.0 / samples_per_pixel as f64;
        r = (scale * r).sqrt();
        g = (scale * g).sqrt();
        b = (scale * b).sqrt();

        // Write the translated [0,255] value of each color component
        ((256.0 * clamp(r, 0.0, 0.999)) as u8,
         (256.0 * clamp(g, 0.0, 0.999)) as u8,
         (256.0 * clamp(b, 0.0, 0.999)) as u8)
    }
}

// Ray

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(orig: &Point3, dir: &Vec3) -> Self {
        Self {origin: *orig,
              direction: *dir}
    }

    pub fn at(&self, t: f64) -> Point3 {
        self.origin + self.direction * t
    }
}

// hittable

pub struct HitRecord {
    pub p: Point3,
    pub normal: Vec3,
    pub mat_ptr: Box<dyn material>,
    pub t: f64,
    pub front_face: bool,
}

impl HitRecord {
    pub fn set_face_normal(&mut self, r: &Ray, outward_normal: &Vec3) {
        self.front_face = Vec3::dot(&r.direction, outward_normal) < 0.0;

        self.normal = match self.front_face {
            true => *outward_normal,
            false => -*outward_normal,
        };
    }

    pub fn default() -> Self {
        Self {p: Point3::new(0.0, 0.0, 0.0),
              normal: Vec3::new(0.0, 0.0, 0.0),
              mat_ptr: <dyn material>::default(),
              t: 0.0,
              front_face: false}
    }

    pub fn copy_rec(&self) -> Self {
        Self {p: self.p,
              normal: self.normal,
              mat_ptr: self.mat_ptr.box_clone(),
              t: self.t,
              front_face: self.front_face}
    }
}

pub trait Hittable : Send {
    fn hit (&self, r: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool {
        false
    }
}

// Sphere Implementation

pub struct Sphere {
    center: Point3,
    radius: f64,
    mat_ptr: Box<dyn material>
}

impl Sphere {
    pub fn new(cen: Point3, r: f64, m: Box<dyn material>) -> Self {
        Self {center: cen,
              radius: r,
              mat_ptr: m}
    }
}

impl Hittable for Sphere {
    fn hit (&self, r: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool {
        let oc: Vec3 = r.origin - self.center;
        let a = r.direction.length_squared();
        let half_b = Vec3::dot(&oc, &r.direction);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return false;
        }
        let sqrtd = discriminant.sqrt();

        // Find the nearest root that lies in the acceptable range
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return false;
            }
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        let outward_normal: Vec3 = (rec.p - self.center) / self.radius;
        rec.set_face_normal(r, &outward_normal);
        rec.mat_ptr = self.mat_ptr.clone();

        true
    }
}

// Storage for hittable objects
#[derive(Default)]
pub struct HittableList {
    pub objects: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    pub fn new(object: Box<dyn Hittable>) -> Self {
        Self {objects: vec![object]}
    }

    pub fn clear(&mut self) {
        self.objects.clear();
    }

    pub fn add(&mut self, object: Box<dyn Hittable>) {
        self.objects.push(object);
    }
}



impl Hittable for HittableList {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool {
        let mut temp_rec: HitRecord = HitRecord::default();
        let mut hit_anything = false;
        let mut closest_so_far = t_max;

        for object in &self.objects {
            if object.hit(r, t_min, closest_so_far, &mut temp_rec) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                *rec = temp_rec.copy_rec();
            }
        }

        hit_anything
    }
}

// Constants
pub const INF: f64 = f64::INFINITY;
pub const PI: f64 = std::f64::consts::PI;

// Utility Functions
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

pub fn random_f64() -> f64{
    let mut rng = rand::thread_rng();
    rng.gen::<f64>()
}

pub fn random_f64_rng(min: f64, max: f64) -> f64 {
    min + (max - min) * random_f64()
}

// Camera Implementation
#[derive(Default)]
pub struct Camera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3
}

impl Camera {
    pub fn new(lookFrom: Point3, lookAt: Point3, vup: Vec3, vfov: f64, aspect_ratio: f64) -> Self {
        let theta = degrees_to_radians(vfov);
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookFrom - lookAt).unit_vector();
        let u = Vec3::cross(&vup, &w);
        let v = Vec3::cross(&w, &u);

        let orig = lookFrom;
        let hor = u * viewport_width;
        let ver = v * viewport_height;
        let llc = orig - hor / 2.0 - ver / 2.0 - w;

        Self {
            origin: orig,
            horizontal: hor,
            vertical: ver,
            lower_left_corner: llc,
        }
    }

    pub fn get_ray(&self, s: f64, t: f64) -> Ray {
        Ray::new(&self.origin, &(self.lower_left_corner + self.horizontal * s + self.vertical * t - self.origin))
    }
}

// Function for color
pub fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x < min {min}
    else if x > max {max}
    else {x}
}

// Begin materials
pub trait material : Send {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, attenuation: &mut Color, scattered: &mut Ray) -> bool {
        false
    }

    fn box_clone(&self) -> Box<dyn material> {
        <dyn material>::default()
    }
}

impl dyn material {
    pub fn new_frm_mat(mat: &str, x: f64, y: f64, z: f64, f: f64) -> Box<Self>{
        match mat {
            "Lambertian" => Box::new(Lambertian::new(&Color::new(x, y, z))),
            "Metal" => Box::new(Metal::new(&Color::new(x, y, z), f)),
            "Dielectric" => Box::new(Dielectric::new(f)),
            _ => Box::new(Lambertian::new(&Color::new(x, y, z))),
        }
    }

    fn default() -> Box<Self> {
        Box::new(Lambertian::new(&Color::new(0.0, 0.0, 0.0)))
    }
}

impl Clone for Box<dyn material> {
    fn clone(&self) -> Box<dyn material> {
        self.box_clone()
    }
}

#[derive(Clone)]
pub struct Lambertian {
    albedo: Color
}

impl Lambertian {
    pub fn new(a: &Color) -> Self {
        Self {albedo: *a}
    }
}

impl material for Lambertian {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, attenuation: &mut Color, scattered: &mut Ray) -> bool {
        let mut scatter_direction = rec.normal + random_unit_vector();

        // Catch degenerate scatter direction
        if scatter_direction.near_zero() {
            scatter_direction = rec.normal;
        }

        *scattered = Ray::new(&rec.p, &scatter_direction);
        *attenuation  = self.albedo;

        true
    }

    fn box_clone(&self) -> Box<dyn material> {
        Box::new((*self).clone())
    }
}

#[derive(Clone)]
pub struct Metal {
    albedo: Color,
    fuzz: f64
}

impl Metal {
    pub fn new(a: &Color, f: f64) -> Self {
        Self {albedo: *a,
              fuzz: f}
    }
}

impl material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, attenuation: &mut Color, scattered: &mut Ray) -> bool {
        let mut reflected = Vec3::reflect(&r_in.direction.unit_vector(), &rec.normal);
        *scattered = Ray::new(&rec.p, &(reflected + random_in_unit_sphere() * self.fuzz));
        *attenuation = self.albedo;
        Vec3::dot(&scattered.direction, &rec.normal) > 0.0
    }

    fn box_clone(&self) -> Box<dyn material> {
        Box::new((*self).clone())
    }
}

#[derive(Default, Clone)]
pub struct Dielectric {
    ir: f64 // index of refraction
}

impl Dielectric {
    pub fn new(val: f64) -> Self {
        Self {ir: val}
    }
    fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
        // Use Schlick's approximation for reflectance
        let mut r0: f64 = (1.0 - ref_idx) / (1.0 + ref_idx);
        r0 = r0 * r0;
        r0 + (1.0 - r0) * f64::powf(1.0 - cosine, 5.0)
    }
}

impl material for Dielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, attenuation: &mut Color, scattered: &mut Ray) -> bool {
        *attenuation = Color::new(1.0, 1.0, 1.0);
        let refraction_ratio = match rec.front_face {
            true => (1.0/self.ir),
            false => self.ir,
        };

        let unit_direction = r_in.direction.unit_vector();
        let dot_val = Vec3::dot(&(-unit_direction), &rec.normal);
        let cos_theta = match dot_val.partial_cmp(&1.0).unwrap() {
            cmp::Ordering::Less => dot_val,
            _ => 1.0,
        };

        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = (refraction_ratio * sin_theta) > 1.0;

        let mut direction: Vec3;
        if cannot_refract || Dielectric::reflectance(cos_theta, refraction_ratio) > random_f64() {
            direction = Vec3::reflect(&unit_direction, &rec.normal);
        } else {
            direction = refract(&unit_direction, &rec.normal, refraction_ratio);
        }

        *scattered = Ray::new(&rec.p, &direction);

        true
    }

    fn box_clone(&self) -> Box<dyn material> {
        Box::new((*self).clone())
    }
}
