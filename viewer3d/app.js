import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const status = document.querySelector("#status");
const trajectory = await fetch("./trajectory.json").then((response) => response.json());

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe6eaec);
scene.fog = new THREE.Fog(0xe6eaec, 6, 12);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 80);
camera.position.set(2.9, -5.0, 2.4);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.65);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.48;

scene.add(new THREE.HemisphereLight(0xf9fbff, 0x9aa0a2, 2.1));

const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
keyLight.position.set(-3, -4, 6);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xb8d2ff, 0.9);
fillLight.position.set(3, 2, 3);
scene.add(fillLight);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(8, 2.8),
  new THREE.MeshStandardMaterial({ color: 0xd5dad8, roughness: 0.85 }),
);
floor.rotation.x = -Math.PI / 2;
floor.position.z = -0.09;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(8, 32, 0x7e878a, 0xb4bbbd);
grid.rotation.x = Math.PI / 2;
grid.position.z = -0.085;
scene.add(grid);

const railMaterial = new THREE.MeshStandardMaterial({ color: 0x1c2023, roughness: 0.42, metalness: 0.2 });
const railGeometry = new THREE.BoxGeometry(6.2, 0.035, 0.04);
for (const y of [-0.26, 0.26]) {
  const rail = new THREE.Mesh(railGeometry, railMaterial);
  rail.position.set(0, y, 0.02);
  rail.castShadow = true;
  scene.add(rail);
}

for (const x of [-2.4, 0, 2.4]) {
  const marker = new THREE.Mesh(
    new THREE.BoxGeometry(0.025, 0.7, 0.012),
    new THREE.MeshStandardMaterial({ color: x === 0 ? 0x23282b : 0xb73b30 }),
  );
  marker.position.set(x, 0, -0.05);
  scene.add(marker);
}

const cart = new THREE.Group();
scene.add(cart);

const cartBody = new THREE.Mesh(
  new THREE.BoxGeometry(0.42, 0.34, 0.18),
  new THREE.MeshStandardMaterial({ color: 0x154f8c, roughness: 0.35, metalness: 0.08 }),
);
cartBody.castShadow = true;
cartBody.receiveShadow = true;
cartBody.position.z = 0.09;
cart.add(cartBody);

const cartTop = new THREE.Mesh(
  new THREE.BoxGeometry(0.28, 0.25, 0.045),
  new THREE.MeshStandardMaterial({ color: 0xf1f5f7, roughness: 0.5 }),
);
cartTop.castShadow = true;
cartTop.position.z = 0.205;
cart.add(cartTop);

const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x17191c, roughness: 0.7 });
const wheelGeometry = new THREE.CylinderGeometry(0.055, 0.055, 0.05, 32);
for (const x of [-0.14, 0.14]) {
  for (const y of [-0.22, 0.22]) {
    const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
    wheel.rotation.x = Math.PI / 2;
    wheel.position.set(x, y, -0.02);
    wheel.castShadow = true;
    cart.add(wheel);
  }
}

const pivot = new THREE.Mesh(
  new THREE.SphereGeometry(0.06, 32, 16),
  new THREE.MeshStandardMaterial({ color: 0xa9b0b4, roughness: 0.28, metalness: 0.5 }),
);
pivot.position.z = 0.18;
pivot.castShadow = true;
cart.add(pivot);

const linkColors = [0xd34b35, 0x26925e, 0xe5ac2f, 0x7553aa];
const linkGroups = [];
for (let index = 0; index < trajectory.links; index += 1) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color: linkColors[index],
    roughness: 0.38,
    metalness: 0.05,
  });

  const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, trajectory.linkLength, 32), material);
  rod.rotation.x = Math.PI / 2;
  rod.position.z = trajectory.linkLength / 2;
  rod.castShadow = true;
  group.add(rod);

  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.052, 32, 16), material);
  tip.position.z = trajectory.linkLength;
  tip.castShadow = true;
  group.add(tip);

  scene.add(group);
  linkGroups.push(group);
}

const resetFlash = new THREE.Mesh(
  new THREE.RingGeometry(0.18, 0.23, 48),
  new THREE.MeshBasicMaterial({ color: 0x1c78be, transparent: true, opacity: 0 }),
);
resetFlash.rotation.x = Math.PI / 2;
resetFlash.position.z = 0.02;
scene.add(resetFlash);

let frameIndex = 0;
const frames = trajectory.frames;
status.textContent = `${trajectory.control}, ${trajectory.links} link${trajectory.links === 1 ? "" : "s"}`;

function updateMechanism(frame) {
  cart.position.x = frame.cartX;
  cart.position.y = 0;
  cart.position.z = 0;

  let currentX = frame.cartX;
  let currentZ = 0.18;
  let cumulativeAngle = 0;
  for (let index = 0; index < trajectory.links; index += 1) {
    cumulativeAngle += frame.angles[index];
    const group = linkGroups[index];
    group.position.set(currentX, 0, currentZ);
    group.rotation.set(0, cumulativeAngle, 0);
    currentX += Math.sin(cumulativeAngle) * trajectory.linkLength;
    currentZ += Math.cos(cumulativeAngle) * trajectory.linkLength;
  }

  resetFlash.position.x = frame.cartX;
  resetFlash.material.opacity = frame.reset ? 0.55 : Math.max(0, resetFlash.material.opacity - 0.04);
}

function animate() {
  requestAnimationFrame(animate);
  updateMechanism(frames[frameIndex]);
  frameIndex = (frameIndex + 1) % frames.length;
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();

