use indicatif::{MultiProgress, ProgressBar, ProgressStyle, ProgressIterator};
use std::env;
use std::str::FromStr;
use std::sync::mpsc::{channel};
use threadpool::ThreadPool;
use image::{RgbImage, Rgb, ImageBuffer};
use ray_tracing::*;

fn parse_input<T: FromStr>(s: &str) -> Option<T> {
    match T::from_str(&s) {
        Ok(x) => Some(x),
        _ => None,
    }
}

fn ray_color(r: &Ray, world: &dyn Hittable, depth: i32) -> Color {
    let mut rec: HitRecord = HitRecord::default();

    // If we've exceeded the ray bounce limit, no more light is gathered
    if depth <= 0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    if world.hit(r, 0.001, INF, &mut rec) {
        let mut scattered: Ray = Default::default();
        let mut attenuation: Color = Default::default();

        let mat = rec.mat_ptr.box_clone();

        if mat.scatter(r, &mut rec, &mut attenuation, &mut scattered) {
            return attenuation * ray_color(&scattered, world, depth - 1);
        }
        return Color::new(0.0, 0.0, 0.0);
    }

    let unit_direction: Vec3 = r.direction.unit_vector();
    let t = 0.5 * (unit_direction.y +  1.0);
    Color::new(1.0, 1.0, 1.0) * (1.0 - t) + Color::new(0.5, 0.7, 1.0) * t
}

fn render(image_height: u32,
          image_width: u32,
          scene: u8,
          samples_per_pixel: u32,
          max_depth: i32) -> Result<(), Box<dyn std::error::Error>> {

    let aspect_ratio = image_width as f64 / image_width as f64;

    let mut img = ImageBuffer::new(image_width, image_height);

    let pool = ThreadPool::new(num_cpus::get());
    let (tx, rx) = channel();

    for j in 0..image_height {
        let tx = tx.clone();

        let mut cam: Camera = Default::default();
        let mut world: HittableList = Default::default();

//         let mut material_ground: Lambertian;
//         let mut material_center: Lambertian;
//         let mut material_left: Lambertian;
//         let mut material_right: Lambertian;


        if scene == 1 {
            cam = Camera::new(Point3::new(-2.0, 2.0, 1.0), Point3::new(0.0, 0.0, -1.0),
                                Vec3::new(0.0, 1.1, 0.0), 90.0, aspect_ratio);
            let material_ground = <dyn material>::new_frm_mat("Lambertian", 0.8, 0.8, 0.0, 0.0);
            let material_center = <dyn material>::new_frm_mat("Lambertian", 0.1, 0.2, 0.5, 0.0);
            let material_left =   <dyn material>::new_frm_mat("Dielectric", 0.0, 0.0, 0.0, 1.5);
            let material_right =  <dyn material>::new_frm_mat("Metal",      0.8, 0.6, 0.2, 0.0);

            world.add(Box::new(Sphere::new(Point3::new( 0.0, -100.5, -1.0), 100.0, material_ground)));
            world.add(Box::new(Sphere::new(Point3::new( 0.0,    0.0, -1.0),   0.5, material_center)));
            world.add(Box::new(Sphere::new(Point3::new(-1.0,    0.0, -1.0),   0.5, material_left.clone())));
            world.add(Box::new(Sphere::new(Point3::new(-1.0,    0.0, -1.0), -0.45, material_left)));
            world.add(Box::new(Sphere::new(Point3::new( 1.0,    0.0, -1.0),   0.5, material_right)));
        } else if scene == 2 {
            cam = Camera::new(Point3::new(0.0, 0.0, 1.0), Point3::new(0.0, 0.0, -1.0),
                                    Vec3::new(0.0, 1.0, 0.0), 90.0, aspect_ratio);

            let material_ground = <dyn material>::new_frm_mat("Lambertian", 0.8, 0.8, 0.0, 0.0);
            let material_center = <dyn material>::new_frm_mat("Lambertian", 0.1, 0.2, 0.5, 0.0);
            let material_left =   <dyn material>::new_frm_mat("Dielectric", 0.0, 0.0, 0.0, 1.5);
            let material_right =  <dyn material>::new_frm_mat("Metal",      0.8, 0.6, 0.2, 0.0);

            world.add(Box::new(Sphere::new(Point3::new( 0.0, -100.5, -1.0), 100.0, material_ground)));
            world.add(Box::new(Sphere::new(Point3::new( 0.0,    0.0, -1.0),   0.5, material_center)));
            world.add(Box::new(Sphere::new(Point3::new(-1.0,    0.0, -1.0),   0.5, material_left.clone())));
            world.add(Box::new(Sphere::new(Point3::new(-1.0,    0.0, -1.0), -0.45, material_left)));
            world.add(Box::new(Sphere::new(Point3::new( 1.0,    0.0, -1.0),   0.5, material_right)));
        }


//         let material_ground = Box::new(Lambertian::new(&Color::new(0.8, 0.8, 0.0)));



//         world.add(Box::new(Sphere::new(Point3::new( 1.0,    0.0, -1.0),   0.5, material_right)));

        pool.execute(move || for i in (0..image_width).progress() {
            let mut pixel_color: Color = Color::new(0.0, 0.0, 0.0);
            for _s in 0..samples_per_pixel {
                let u = (i as f64 + random_f64()) / (image_width - 1) as f64;
                let v = (j as f64 + random_f64()) / (image_height - 1) as f64;
                let r: Ray = cam.get_ray(u, v);
                pixel_color += ray_color(&r, &world, max_depth);
            }
            tx.send((i, (image_height - 1) - j, pixel_color.write_color(samples_per_pixel))).expect("Could not send data!");
        });
    }

    for _ in 0..(image_width * image_height) {
        let (i, j, (r, g, b)) = rx.recv()?;
        img.put_pixel(i, j, Rgb([r, g, b]));
    }

    let _ = img.save(&(format!("scene_{}.png", scene)))?;

    Ok(())
}

fn main() {
    // Image

    let args: Vec<String> = env::args().collect();
    let mut image_width: u32 = 400;
    let mut scene = 1;

    if args.len() != 3 && args.len() != 1 {
        eprintln!("Usage: {} IMAGE_WIDTH SCENE", args[0]);
        eprintln!("Example: {} 400 1", args[0]);
        std::process::exit(1);
    } else if args.len() == 3 {
        image_width = parse_input(&args[1]).expect("Error parsing image width.");
        scene = parse_input(&args[2]).expect("Error parsing scene selection.");
    }

    let samples_per_pixel: u32 = 100;
    let max_depth = 50;
    let aspect_ratio = 16.0 / 9.0;
    let image_height: u32 = (image_width as f64 / aspect_ratio) as u32;
    // World

//     let mut world: HittableList = Default::default();
//     world.add(Box::new(Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5)));
//     world.add(Box::new(Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0)));

    // Camera

//     let cam = Camera::new();

    // Render - Multithreaded

    let result = render(image_height, image_width, scene, samples_per_pixel, max_depth);
    match result {
        Err(e) => panic!("{}", e),
        _ => eprint!("\nDone.\n"),
    }
}
