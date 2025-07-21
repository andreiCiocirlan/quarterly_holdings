$(document).ready(function() {
var table = $('#filersTable').DataTable({
    dom: '<"top"f>rt<"bottom"lip><"clear">',
    pageLength: 25,
    order: [[1, 'desc']],  // index 1 is Shares (2nd column)
    columnDefs: [
      { targets: [1], type: 'num-fmt' },   // Use 'num-fmt' for Shares
      { targets: [2], type: 'num-fmt' }    // Value is currency, also 'num-fmt'
    ],
    language: {
      searchPlaceholder: "Filter filers..."
    },
    responsive: true,
    deferRender: true
  });

  // Optional: fixed header while scrolling
  new $.fn.dataTable.FixedHeader(table);
});