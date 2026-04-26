const customers = [];

let currentId = 1;

const form = document.getElementById("customerForm");
const tableBody = document.getElementById("dataTableBody");

const statTotal = document.getElementById("stat-total");
const statStayed = document.getElementById("stat-stayed");
const statExited = document.getElementById("stat-exited");

/* ---------------- LOADER ---------------- */
const loader = document.createElement("div");
loader.id = "loader";
loader.innerHTML = `
<div style="
position:fixed;
top:0;left:0;
width:100%;height:100%;
background:rgba(0,0,0,0.4);
display:none;
align-items:center;
justify-content:center;
z-index:9999;">
    <div style="
    background:white;
    padding:20px 30px;
    border-radius:10px;
    font-weight:bold;">
        Loading...
    </div>
</div>
`;
document.body.appendChild(loader);

function showLoader() {
  loader.firstElementChild.style.display = "flex";
}

function hideLoader() {
  loader.firstElementChild.style.display = "none";
}

/* ---------------- STATS ---------------- */
function updateStats() {
  const total = customers.length;
  const exited = customers.filter((c) => c.prediction === 1).length;

  statTotal.textContent = total;
  statExited.textContent = exited;
  statStayed.textContent = total - exited;
}

/* ---------------- TABLE ---------------- */
function renderTable() {
  const emptyState = document.getElementById("emptyState");

  if (customers.length === 0) {
    emptyState.style.display = "flex";
    tableBody.innerHTML = "";
    return;
  } else {
    emptyState.style.display = "none";
  }

  tableBody.innerHTML = customers
    .map(
      (c) => `
        <tr class="text-sm">
            <td class="px-6 py-4">${c.id}</td>
            <td class="px-6 py-4">${c.surname}</td>
            <td class="px-6 py-4">${c.creditScore}</td>
            <td class="px-6 py-4">${c.geography}</td>
            <td class="px-6 py-4">${c.numProducts}</td>
            <td class="px-6 py-4">
                ${c.isActiveMember ? "Active" : "Inactive"}
            </td>
            <td class="px-6 py-4 font-bold ${
              c.prediction === 1 ? "text-red-500" : "text-green-600"
            }">
              ${
                c.prediction === 1
                  ? `EXIT (${(c.probability * 100).toFixed(1)}%)`
                  : `STAY (${(100 - c.probability * 100).toFixed(1)}%)`
              }
            </td>
            <td class="px-6 py-4">
            <div class="flex items-center space-x-2">
                <div class="w-24 bg-slate-200 rounded-full h-2 overflow-hidden">
                <div 
                    class="h-2 rounded-full ${
                      c.prediction === 1 ? "bg-red-500" : "bg-green-500"
                    }"
                    style="width:${(c.probability * 100).toFixed(2)}%">
                </div>
                </div>
                <span class="text-xs font-semibold text-slate-600">
                ${(c.probability * 100).toFixed(2)}%
                </span>
            </div>
            </td>
        </tr>
    `,
    )
    .join("");
}

/* ---------------- FORM SUBMIT ---------------- */
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const data = {
    id: currentId++, // AUTO INCREMENT FIX
    surname: document.getElementById("surname").value,
    creditScore: Number(document.getElementById("creditScore").value),
    geography: document.getElementById("geography").value,
    gender: document.getElementById("gender").value,
    age: Number(document.getElementById("age").value),
    tenure: Number(document.getElementById("tenure").value),
    numProducts: Number(document.getElementById("numProducts").value),
    balance: Number(document.getElementById("balance").value),
    salary: Number(document.getElementById("salary").value),
    hasCrCard: document.getElementById("hasCrCard").checked ? 1 : 0,
    isActiveMember: document.getElementById("isActive").checked ? 1 : 0,
  };

  try {
    showLoader();

    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = await res.json();

    data.prediction = result.prediction;
    data.probability = result.probability;

    customers.unshift(data);

    renderTable();
    updateStats();

    form.reset();

    // reset default form values (optional improvement)
    document.getElementById("customerId").value = currentId;
  } catch (error) {
    console.error("Error:", error);
    alert("Prediction failed. Check backend.");
  } finally {
    hideLoader();
  }
});
