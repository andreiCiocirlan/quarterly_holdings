document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('searchInput');
  const dropdown = document.getElementById('resultsDropdown');

  input.addEventListener('input', async () => {
    const query = input.value.trim();
    if (query.length < 2) {
      dropdown.style.display = 'none';
      dropdown.innerHTML = '';
      return;
    }

    try {
      const response = await fetch(`/api/search_filers?q=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error('Network error');

      const filers = await response.json();

      if (filers.length === 0) {
        dropdown.style.display = 'none';
        dropdown.innerHTML = '';
        return;
      }

      dropdown.innerHTML = filers.map(filer => `
        <li class="list-group-item" data-cik="${filer.cik}">
          ${filer.formatted_name.replace(/_/g, ' ')}
        </li>
      `).join('');
      dropdown.style.display = 'block';

    } catch (error) {
      console.error('Error fetching filers:', error);
      dropdown.style.display = 'none';
      dropdown.innerHTML = '';
    }
  });

  dropdown.addEventListener('click', e => {
    if (e.target.tagName.toLowerCase() === 'li') {
      const cik = e.target.getAttribute('data-cik');
      const name = e.target.textContent.trim().toLowerCase().replace(/ /g, '-').replace(/[^a-z0-9\-]/g, '');
      window.location.href = `/manager/${cik}-${name}`;
    }
  });

  document.addEventListener('click', e => {
    if (!input.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.style.display = 'none';
    }
  });
});