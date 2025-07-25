

$(document).ready(function() {
  var table = $('#filersAumTable').DataTable({
    pageLength: 50,
    order: [[2, 'desc']],  // Sort by Holdings Value column (index 2 zero-based)
    columnDefs: [
      { targets: [1], type: 'num-fmt' },  // Holdings Count
      { targets: [2], type: 'formatted-num' }   // Holdings Value
    ],
    language: {
      searchPlaceholder: "Filter filers..."
    },
    responsive: true,
    dom: '<"top"f>rt<"bottom"lip><"clear">',
    deferRender: true
  });

  // Optional: fixed header while scrolling
  new $.fn.dataTable.FixedHeader(table);
});