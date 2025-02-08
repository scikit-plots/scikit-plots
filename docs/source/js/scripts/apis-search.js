/**
 * This script is for initializing the search table on the API index page. See
 * DataTables documentation for more information: https://datatables.net/
 */

document.addEventListener("DOMContentLoaded", function () {
  // Ensure DataTable is initialized only after the table is available
  const table = document.querySelector("table.apis-search-table");
  if (table) {
    console.log("Initializing DataTable on:", table);
    const dataTable = new DataTable(table, {
      order: [], // Keep original order
      lengthMenu: [10, 25, 50, 100, { label: "All", value: -1 }],
      pageLength: -1, // Show all entries by default
    });
    // dataTable.draw(); // Force redraw
  } else {
    console.error("Table with class 'apis-search-table' not found.");
  }
});


// jQuery(document).ready(function() {
//   var table = jQuery(".apis-search-table");
//   console.log(table);  // Check if the table is selected
//   if (table.length) {
//     table.DataTable({
//       order: [],
//       lengthMenu: [10, 25, 50, 100, { label: "All", value: -1 }],
//       pageLength: -1,
//     });
//   }
// });
