use std::{env, fs, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    copy_model(
        &manifest_dir,
        "G_MAP_SE_BURN_RS",
        "../burn_models/g_map_se/g_map_se.rs",
        "g_map_se.rs",
        "uv run prepare_burn_models.py",
    );
    copy_model(
        &manifest_dir,
        "ECAPA_BURN_RS",
        "../burn_models/voxceleb_ecapa512/voxceleb_ecapa512.rs",
        "voxceleb_ecapa512.rs",
        "uv run prepare_burn_models.py",
    );
}

fn copy_model(
    manifest_dir: &PathBuf,
    env_var: &str,
    default_path: &str,
    destination_name: &str,
    generate_command: &str,
) {
    let model_rs = env::var_os(env_var)
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join(default_path));
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let destination = out_dir.join(destination_name);

    println!("cargo:rerun-if-env-changed={env_var}");
    println!("cargo:rerun-if-changed={}", model_rs.display());

    if let Err(error) = fs::copy(&model_rs, &destination) {
        let message = format!(
            "Unable to copy generated Burn model source from '{}': {error}\n\
             Generate it first with:\n\
             {generate_command}\n\
             Or set {env_var}=/path/to/generated_model.rs",
            model_rs.display()
        );
        fs::write(&destination, format!("compile_error!({message:?});\n")).unwrap();
    }
}
