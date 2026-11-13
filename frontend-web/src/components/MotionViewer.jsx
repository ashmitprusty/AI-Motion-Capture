import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { BVHLoader } from "three/addons/loaders/BVHLoader.js";

/**
 * Phase 5: loads the generated .bvh and plays it on a 3D avatar (spheres + limbs on the skeleton).
 * The BVH is in centimetres, so everything is placed in a group scaled by 0.01 (-> metres).
 */
export default function MotionViewer({ url }) {
  const mountRef = useRef(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    const mount = mountRef.current;
    setStatus("loading");
    let disposed = false;
    let raf = 0;
    let mixer = null;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);
    const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / mount.clientHeight, 0.1, 50);
    camera.position.set(0, 1.3, 3.4);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1, 0);
    controls.update();

    scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(2, 4, 3);
    scene.add(sun);
    scene.add(new THREE.GridHelper(8, 16, 0x475569, 0x1e293b));

    const group = new THREE.Group();
    group.scale.setScalar(0.01);
    scene.add(group);

    const sphereGeo = new THREE.SphereGeometry(1, 20, 14);
    const limbGeo = new THREE.CylinderGeometry(1, 1, 1, 12);
    const jointMat = new THREE.MeshStandardMaterial({ color: 0xf43f5e });
    const limbMat = new THREE.MeshStandardMaterial({ color: 0x22d3ee });
    let helper = null;

    new BVHLoader().load(
      url,
      (result) => {
        if (disposed) return;
        const bones = result.skeleton.bones;
        const root = bones[0];
        group.add(root);

        // Limbs: a cylinder from each bone's parent to the bone, attached to the parent so it animates with it.
        bones.forEach((bone) => {
          if (bone.parent && bone.parent.isBone) {
            const offset = bone.position.clone();
            const length = offset.length();
            if (length > 0.001) {
              const limb = new THREE.Mesh(limbGeo, limbMat);
              limb.scale.set(2.5, length, 2.5);
              limb.position.copy(offset).multiplyScalar(0.5);
              limb.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), offset.clone().normalize());
              bone.parent.add(limb);
            }
          }
          const joint = new THREE.Mesh(sphereGeo, jointMat);
          joint.scale.setScalar(bone.name === "Head" ? 11 : 3.5);
          bone.add(joint);
        });

        helper = new THREE.SkeletonHelper(root);
        scene.add(helper);

        mixer = new THREE.AnimationMixer(root);
        mixer.clipAction(result.clip).play(); // loops forever
        setStatus("ready");
      },
      undefined,
      () => !disposed && setStatus("error"),
    );

    const clock = new THREE.Clock();
    const animate = () => {
      raf = requestAnimationFrame(animate);
      if (mixer) mixer.update(clock.getDelta());
      renderer.render(scene, camera);
    };
    animate();

    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      observer.disconnect();
      controls.dispose();
      sphereGeo.dispose();
      limbGeo.dispose();
      jointMat.dispose();
      limbMat.dispose();
      if (helper) helper.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [url]);

  return (
    <div className="viewer" ref={mountRef}>
      {status === "loading" && <div className="viewer-msg">Loading animation…</div>}
      {status === "error" && <div className="viewer-msg warn">Could not load the animation file.</div>}
    </div>
  );
}
