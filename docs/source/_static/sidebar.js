// The navigation tree is fully expanded, so scroll the current page into view in the sidebar.
window.addEventListener('load', () => {
  const sidebar = document.querySelector('.sidebar-scroll');
  const current = document.querySelector('.sidebar-tree .current-page > .reference');
  if (!sidebar || !current) return;
  const offset = current.getBoundingClientRect().top - sidebar.getBoundingClientRect().top;
  sidebar.scrollTop += offset - sidebar.clientHeight / 3;
});
