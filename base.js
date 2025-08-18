// 로딩 스크린
window.addEventListener('load', () => {
  const loading = document.getElementById('loadingScreen');
  if (!loading) return;

  // 최소 0.5초 동안 로딩화면 유지
  setTimeout(() => {
    loading.classList.add('hidden'); // 마스크 이동 시작
    // 마스크 애니메이션 끝나면 완전히 제거
    setTimeout(() => {
      loading.style.display = 'none';
    }, 1500); // 마스크 transition 시간과 맞추기
  }, 500); // 0.5초 유지
});

// 커서
const cursor = document.getElementById('cursor');
const hoverElements = document.querySelectorAll('a, input, button');
const dateSelectElements = document.querySelectorAll('.signup-form input[type="date"], select');

let isHidingCursor = false; // 클릭 상태 체크

// hover 시 커서 이미지 변경
hoverElements.forEach(el => {
  el.addEventListener('mouseenter', () => cursor.classList.add('hover'));
  el.addEventListener('mouseleave', () => cursor.classList.remove('hover'));
});

// 클릭 시 커스텀 커서 숨기고 기본 커서 표시
dateSelectElements.forEach(el => {
  el.addEventListener('mouseenter', () => {
    cursor.style.display = 'none';
    document.body.style.cursor = 'auto';
  });
  el.addEventListener('mouseleave', () => {
    cursor.style.display = 'block';
    document.body.style.cursor = 'none';
  });
});

// iframe 기존 방식 유지
const iframeElements = document.querySelectorAll('iframe');
iframeElements.forEach(el => {
  el.addEventListener('mouseenter', () => {
    cursor.style.display = 'none';
    document.body.style.cursor = 'auto';
  });
  el.addEventListener('mouseleave', () => {
    cursor.style.display = 'block';
    document.body.style.cursor = 'none';
  });
});

// 커서 이동
document.addEventListener('mousemove', e => {
  if (!isHidingCursor) {
    cursor.style.left = e.clientX + 'px';
    cursor.style.top = e.clientY + 'px';
  }
});







// 파티클
document.addEventListener('DOMContentLoaded', () => {
  const particlesContainer = document.getElementById('particles');
  const particleCount = 10;
  const particleImages = [
    '../static/images/fried-egg.png',
    '../static/images/avocado.png',
    '../static/images/meat.png',
    '../static/images/frying-pan.png',
    '../static/images/fish.png'
  ];

  for (let i = 0; i < particleCount; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';

    particle.style.left = Math.random() * 100 + '%';
    particle.style.top = (Math.random() * -20) + 'vh';

    const size = Math.random() * 20 + 30;  // 30 ~ 50 px
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;

    particle.style.backgroundImage = `url(${particleImages[Math.floor(Math.random() * particleImages.length)]})`;

    particle.style.opacity = '1';

    particle.style.animationDuration = (Math.random() * 5 + 7) + 's';
    particle.style.animationDelay = (Math.random() * 10) + 's';

    particlesContainer.appendChild(particle);
  }
});


// 통통 튀는 스티커
const popElements = document.querySelectorAll(
  'footer .pop-avocado, footer .pop-meat, footer .pop-pan, footer .pop-friedegg, footer .pop-bacon'
);

popElements.forEach(el => {
  let posX = 0, posY = 0;
  let velocityX = 0, velocityY = 0;
  let targetX = 0, targetY = 0;
  const spring = 0.25;
  const friction = 0.6;
  const maxDistance = 120;

  // CSS에서 설정한 초기 회전값 가져오기
  const initialTransform = window.getComputedStyle(el).transform;
  let initialRotate = 0;
  if (initialTransform !== 'none') {
    const matrix = new DOMMatrix(initialTransform);
    initialRotate = Math.atan2(matrix.b, matrix.a) * (180 / Math.PI);
  }

  const rect = el.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;

  el.addEventListener('mousemove', (e) => {
    const deltaX = e.clientX - centerX;
    const deltaY = e.clientY - centerY;

    targetX = Math.max(-maxDistance, Math.min(maxDistance, -deltaX * 0.8));
    targetY = Math.max(-maxDistance, Math.min(maxDistance, -deltaY * 0.8));
  });

  el.addEventListener('mouseleave', () => {
    targetX = 0;
    targetY = 0;
  });

  function animate() {
    const dx = targetX - posX;
    const dy = targetY - posY;

    velocityX += dx * spring;
    velocityY += dy * spring;

    velocityX *= friction;
    velocityY *= friction;

    posX += velocityX;
    posY += velocityY;

    // 기존 회전값 유지하면서 translate/scale 적용
    el.style.transform = `translate(${posX}px, ${posY}px) rotate(${initialRotate}deg) scale(1.1)`;

    requestAnimationFrame(animate);
  }
  animate();
});

