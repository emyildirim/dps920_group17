"""Connect the trained steering model to the Udacity simulator."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path

import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from tensorflow.keras.models import load_model

from preprocessing import preprocess_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model.h5 in Udacity Autonomous Mode."
    )
    parser.add_argument("--model", type=Path, default=Path("model.h5"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4567)
    parser.add_argument("--target-speed", type=float, default=6.0)
    parser.add_argument("--minimum-speed", type=float, default=4.0)
    parser.add_argument(
        "--steering-smoothing",
        type=float,
        default=0.15,
        help="Fraction of the previous command to retain (0 disables smoothing)",
    )
    parser.add_argument("--throttle-gain", type=float, default=0.08)
    parser.add_argument("--base-throttle", type=float, default=0.05)
    parser.add_argument(
        "--turn-speed-reduction",
        type=float,
        default=3.0,
        help="Reduce desired speed by this value times absolute steering",
    )
    args = parser.parse_args()

    if args.port < 1 or args.port > 65535:
        parser.error("--port must be in the range 1-65535")
    if args.target_speed <= 0 or args.minimum_speed < 0:
        parser.error("Speeds must be positive")
    if args.minimum_speed > args.target_speed:
        parser.error("--minimum-speed cannot exceed --target-speed")
    if not 0 <= args.steering_smoothing < 1:
        parser.error("--steering-smoothing must be in the range [0, 1)")
    return args


class DrivingController:
    def __init__(self, model, args: argparse.Namespace) -> None:
        self.model = model
        self.target_speed = args.target_speed
        self.minimum_speed = args.minimum_speed
        self.steering_smoothing = args.steering_smoothing
        self.throttle_gain = args.throttle_gain
        self.base_throttle = args.base_throttle
        self.turn_speed_reduction = args.turn_speed_reduction
        self.previous_steering: float | None = None
        self.frame_count = 0

    def predict_steering(self, image_rgb: np.ndarray) -> tuple[float, float]:
        model_input = np.expand_dims(preprocess_image(image_rgb), axis=0)
        raw_steering = float(
            np.asarray(self.model(model_input, training=False)).reshape(-1)[0]
        )
        raw_steering = float(np.clip(raw_steering, -1.0, 1.0))

        if self.previous_steering is None:
            steering = raw_steering
        else:
            steering = (
                self.steering_smoothing * self.previous_steering
                + (1.0 - self.steering_smoothing) * raw_steering
            )
        self.previous_steering = steering
        return raw_steering, steering

    def calculate_throttle(self, speed: float, steering: float) -> float:
        desired_speed = max(
            self.minimum_speed,
            self.target_speed - self.turn_speed_reduction * abs(steering),
        )
        throttle = self.base_throttle + self.throttle_gain * (desired_speed - speed)
        if speed < self.minimum_speed:
            throttle = max(throttle, 0.25)
        return float(np.clip(throttle, 0.0, 1.0))


def decode_simulator_image(encoded_image: str) -> np.ndarray:
    if "," in encoded_image:
        encoded_image = encoded_image.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded_image)
    with Image.open(BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"))


def create_application(controller: DrivingController):
    sio = socketio.Server(async_mode="eventlet")

    def send_control(sid: str, steering: float, throttle: float) -> None:
        sio.emit(
            "steer",
            data={
                "steering_angle": f"{steering:.8f}",
                "throttle": f"{throttle:.8f}",
            },
            room=sid,
        )

    @sio.on("connect")
    def connect(sid, environ) -> None:
        del environ
        print(f"Simulator connected: {sid}")
        controller.previous_steering = None
        send_control(sid, 0.0, 0.0)

    @sio.on("disconnect")
    def disconnect(sid) -> None:
        print(f"Simulator disconnected: {sid}")

    @sio.on("telemetry")
    def telemetry(sid, data) -> None:
        if not data:
            sio.emit("manual", data={}, room=sid)
            return

        try:
            speed = float(data["speed"])
            image_rgb = decode_simulator_image(data["image"])
            raw_steering, steering = controller.predict_steering(image_rgb)
            throttle = controller.calculate_throttle(speed, steering)
        except (KeyError, TypeError, ValueError, OSError) as error:
            print(f"Could not process telemetry: {error}")
            send_control(sid, 0.0, 0.0)
            return

        send_control(sid, steering, throttle)
        controller.frame_count += 1
        if controller.frame_count % 30 == 0:
            print(
                f"frame={controller.frame_count} speed={speed:.2f} "
                f"raw={raw_steering:+.4f} steering={steering:+.4f} "
                f"throttle={throttle:.3f}"
            )

    return socketio.WSGIApp(sio)


def main() -> int:
    args = parse_args()
    print(f"Loading model: {args.model}")
    model = load_model(args.model, compile=False)
    controller = DrivingController(model, args)
    application = create_application(controller)

    print(f"Listening on {args.host}:{args.port}")
    print("Start the Udacity simulator and select Autonomous Mode.")
    listener = eventlet.listen((args.host, args.port))
    eventlet.wsgi.server(listener, application)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
