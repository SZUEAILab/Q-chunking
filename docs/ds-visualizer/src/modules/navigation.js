import { onLanguageChange, tx } from "../i18n.js";

export function initNavigation() {
  const navLinks = Array.from(document.querySelectorAll(".shadcn-nav a[href^='#']"));
  const nav = document.querySelector(".shadcn-nav");
  const navSections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);
  const progress = document.getElementById("scrollProgress");
  const backToTop = document.getElementById("backToTop");
  const activeNavLabel = document.getElementById("activeNavLabel");
  let currentActiveId = navSections[0]?.id;

  const updateScrollUi = () => {
    const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
    const ratio = Math.min(1, Math.max(0, window.scrollY / maxScroll));
    if (progress) progress.style.transform = `scaleX(${ratio})`;
    if (backToTop) backToTop.classList.toggle("is-visible", window.scrollY > window.innerHeight * 0.65);
  };

  const updateBackToTopLabel = () => {
    if (backToTop) backToTop.setAttribute("aria-label", tx("返回顶部"));
    if (nav) nav.setAttribute("aria-label", tx("页面章节导航"));
  };

  updateScrollUi();
  updateBackToTopLabel();
  window.addEventListener("scroll", updateScrollUi, { passive: true });
  window.addEventListener("resize", updateScrollUi);
  onLanguageChange(() => {
    updateBackToTopLabel();
    if (currentActiveId) setActiveNavLink(currentActiveId);
  });

  if (backToTop) {
    backToTop.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  if (navLinks.length && navSections.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActiveNavLink(visible.target.id);
      },
      {
        rootMargin: "-18% 0px -64% 0px",
        threshold: [0.08, 0.18, 0.32],
      },
    );

    navSections.forEach((section) => observer.observe(section));
    setActiveNavLink(navSections[0].id);
  }

  function setActiveNavLink(id) {
    currentActiveId = id;
    navLinks.forEach((link) => {
      const isActive = link.getAttribute("href") === `#${id}`;
      link.classList.toggle("is-active", isActive);
      if (isActive) {
        link.setAttribute("aria-current", "location");
        link.scrollIntoView({ block: "nearest", inline: "nearest" });
        if (activeNavLabel) {
          activeNavLabel.textContent = link.querySelector(".nav-label")?.textContent?.trim() || link.textContent.trim();
        }
      } else {
        link.removeAttribute("aria-current");
      }
    });
  }
}
