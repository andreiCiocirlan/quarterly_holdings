
const filterInput = document.getElementById('filterInput');
if (filterInput) {
  document.getElementById('filterInput').addEventListener('input', function() {
    const filterText = this.value.toLowerCase();
    const rows = document.querySelectorAll('#holdingsTable tbody tr');

    rows.forEach(row => {
      const rowText = row.textContent.toLowerCase();
      row.style.display = rowText.includes(filterText) ? '' : 'none';
    });
  });
}

$(document).ready(function() {
  // Handle compareSelect navigation
  const compareSelect = document.getElementById('compareSelect');
  if (compareSelect) {
    compareSelect.addEventListener('change', function () {
      const url = this.value;
      if (url) {
        window.location.href = url;  // Navigate to the selected URL
      }
    });
  }

  var table = $('#holdingsTable').DataTable({
    dom: '<"top"f>rt<"bottom"lip><"clear">',
    pageLength: 50,
    order: [[0, 'asc']], // Default sort by Rank ascending
    ordering: true,
    columnDefs: [
      { targets: [0, 2, 5], type: 'num' },        // Rank, Shares, Change as numeric
      { targets: [3], type: 'num-fmt' },          // Value (currency)
      { targets: [4, 6, 7], type: 'percent' }     // Percentage columns (you may need custom sorting if you have NEW/REMOVED)
    ],
    language: {
      searchPlaceholder: "Filter holdings..."
    },
    dom: '<"d-flex justify-content-start my-3"f>rtip',
    responsive: true,  // Optional: enable responsive extension if included
    deferRender: true  // Improves performance on large tables
  });

  new $.fn.dataTable.FixedHeader(table);
  $('#loadingIndicator').hide();
  $('#holdingsTableContainer').show();
});