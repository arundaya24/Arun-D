import { useEffect, useRef } from "react";
import * as THREE from "three";

export default function LiquidCanvas() {
  const mountRef = useRef(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const isMobileDevice = window.innerWidth < 768 || (navigator.maxTouchPoints && navigator.maxTouchPoints > 2);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.1,
      100
    );
    camera.position.set(0, -1.8, 4.2);
    camera.lookAt(0, 0.2, 0);

    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: !isMobileDevice,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const segments = isMobileDevice ? 48 : 96;
    const geometry = new THREE.PlaneGeometry(6.5, 4.5, segments, segments);

    const uniforms = {
      uTime: { value: 0 },
      uMouse: { value: new THREE.Vector2(0, 0) },
      uScroll: { value: 0 },
    };

    const vertexShader = `
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      varying float vElevation;

      uniform float uTime;
      uniform vec2 uMouse;
      uniform float uScroll;

      void main() {
        vUv = uv;
        vec3 pos = position;
        
        float frequencyA = 1.4;
        float frequencyB = 2.8;
        float speed = uTime * 0.35;
        
        float elevation = sin(pos.x * frequencyA + speed) * cos(pos.y * frequencyA + speed * 0.75) * 0.28;
        elevation += sin(pos.x * frequencyB - speed * 0.5 + uMouse.x * 1.2) * 0.12;
        elevation += cos(pos.y * frequencyB + speed * 0.4 - uMouse.y * 1.2) * 0.12;
        elevation -= uScroll * 0.25;
        
        pos.z += elevation;
        vElevation = elevation;

        float delta = 0.01;
        float elevX = sin((pos.x + delta) * frequencyA + speed) * cos(pos.y * frequencyA + speed * 0.75) * 0.28;
        float elevY = sin(pos.x * frequencyA + speed) * cos((pos.y + delta) * frequencyA + speed * 0.75) * 0.28;
        vec3 tangentX = vec3(delta, 0.0, elevX - elevation);
        vec3 tangentY = vec3(0.0, delta, elevY - elevation);
        vec3 computedNormal = normalize(cross(tangentX, tangentY));

        vNormal = normalMatrix * computedNormal;
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        vViewPosition = -mvPosition.xyz;
        gl_Position = projectionMatrix * mvPosition;
      }
    `;

    const fragmentShader = `
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      varying float vElevation;

      uniform vec2 uMouse;

      void main() {
        vec3 normal = normalize(vNormal);
        vec3 viewDir = normalize(vViewPosition);

        float fresnel = pow(1.0 - max(dot(viewDir, normal), 0.0), 3.2);

        vec3 studioLightDir = normalize(vec3(0.5 + uMouse.x * 0.3, 0.8, 0.7 - uMouse.y * 0.2));
        float diffuse = max(dot(normal, studioLightDir), 0.0);

        vec3 halfDir = normalize(studioLightDir + viewDir);
        float specular = pow(max(dot(normal, halfDir), 0.0), 32.0) * 0.45;

        vec3 basePearl = vec3(0.94, 0.96, 0.98);
        vec3 rimSoft = vec3(0.85, 0.90, 0.96);
        vec3 warmSpecular = vec3(0.99, 0.88, 0.76);

        vec3 finalColor = basePearl + (rimSoft * fresnel * 0.45) + (diffuse * 0.05) + (warmSpecular * specular * 0.35);

        float edgeFade = smoothstep(0.0, 0.35, vUv.x) * smoothstep(1.0, 0.65, vUv.x) *
                         smoothstep(0.0, 0.35, vUv.y) * smoothstep(1.0, 0.65, vUv.y);

        float alpha = clamp(edgeFade * (0.28 + fresnel * 0.22), 0.0, 0.55);

        gl_FragColor = vec4(finalColor, alpha);
      }
    `;

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI * 0.32;
    scene.add(mesh);

    let targetMouseX = 0;
    let targetMouseY = 0;
    let currentMouseX = 0;
    let currentMouseY = 0;

    let targetScroll = 0;
    let currentScroll = 0;

    function handleMouseMove(event) {
      targetMouseX = (event.clientX / window.innerWidth) * 2 - 1;
      targetMouseY = -(event.clientY / window.innerHeight) * 2 + 1;
    }

    function handleScroll() {
      const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      targetScroll = window.scrollY / maxScroll;
    }

    window.addEventListener("mousemove", handleMouseMove, { passive: true });
    window.addEventListener("scroll", handleScroll, { passive: true });

    function handleResize() {
      const width = window.innerWidth;
      const height = window.innerHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    }

    window.addEventListener("resize", handleResize);

    const clock = new THREE.Clock();
    let animationFrameId;

    function animate() {
      animationFrameId = requestAnimationFrame(animate);

      const elapsedTime = clock.getElapsedTime();
      uniforms.uTime.value = elapsedTime;

      currentMouseX += (targetMouseX - currentMouseX) * 0.04;
      currentMouseY += (targetMouseY - currentMouseY) * 0.04;
      uniforms.uMouse.value.set(currentMouseX, currentMouseY);

      currentScroll += (targetScroll - currentScroll) * 0.04;
      uniforms.uScroll.value = currentScroll;

      mesh.rotation.z = currentMouseX * 0.08;
      mesh.rotation.y = currentMouseY * 0.05;

      renderer.render(scene, camera);
    }

    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleResize);

      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, []);

  return <div ref={mountRef} className="webgl-layer" aria-hidden="true" />;
}
