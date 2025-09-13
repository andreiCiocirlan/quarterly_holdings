$(document).ready(function() {
var table = $('#latestFilingsTable').DataTable({
    dom: '<"top"f>rt<"bottom"lip><"clear">',
    pageLength: 50,
    order: [[4, 'desc']],  // index 4 is filing_date (5th column)
    columnDefs: [
      { targets: [2], type: 'num-fmt' },        // Use 'num-fmt' for Num holdings
      { targets: [3], type: 'formatted-num' }   // Use 'formatted-num' for Holdings Value
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