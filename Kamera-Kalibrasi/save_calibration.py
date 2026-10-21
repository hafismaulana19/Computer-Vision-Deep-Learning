import numpy as np
from calibration_utils import calibrate_images

calibration = calibrate_images(
    "images/*.jpg",
    display=False
)

np.savez(
    "calib.npz",
    camera_matrix=calibration.camera_matrix,
    distortion_coefficients=calibration.distortion_coefficients,
    image_size=np.array(calibration.image_size),
    rms_error=calibration.rms_error,
    reprojection_rmse=calibration.reprojection_rmse,
)

print("Calibration saved successfully.")
print("File: calib.npz")
print("Image size:", calibration.image_size)
print("RMS error:", calibration.rms_error)
print("Reprojection RMSE:", calibration.reprojection_rmse)