const MODEL = window.DAVE_MODEL;

function present(value) {
  return value !== undefined && value !== null && String(value).trim() !== "";
}

function numberOrNull(value) {
  if (!present(value)) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function sigmoid(value) {
  const clipped = Math.max(-35, Math.min(35, value));
  return 1 / (1 + Math.exp(-clipped));
}

function getRowValue(row, feature, model) {
  if (model.derived_cols.includes(feature)) {
    return derivedFeatures(row, model)[feature];
  }
  return numberOrNull(row[feature]);
}

function derivedFeatures(row, model) {
  const hasOne = (name) => Number(row[name] || 0) === 1;
  const sumOnes = (cols) => cols.reduce((sum, col) => sum + (hasOne(col) ? 1 : 0), 0);
  const labCount = ["count_hemoglobin_24months", "count_HbA1c_24months", "count_AST_24months", "count_ALT_24months"]
    .map((name) => numberOrNull(row[name]))
    .filter((value) => value !== null)
    .reduce((sum, value) => sum + value, 0);

  return {
    ccw_36_sum: sumOnes(model.ccw_36_cols),
    ccw_60_sum: sumOnes(model.ccw_60_cols),
    selected_ccw_sum: sumOnes(model.selected_conditions),
    has_cardiorenal_condition: [
      "cms_ccw_heart_failure_36",
      "cms_ccw_chronic_kidney_disease_36",
      "cms_ccw_ischemic_heart_disease_36",
      "cms_ccw_diabetes_36",
    ].some(hasOne) ? 1 : 0,
    ambulatory_specialty_sum: [
      "Specialty_internal_medicine",
      "Specialty_nephrology",
      "Specialty_cardiology",
    ].reduce((sum, name) => sum + (hasOne(name) ? 1 : 0), 0),
    lab_count_sum_targeted: labCount,
    has_any_target_lab: labCount > 0 ? 1 : 0,
  };
}

function transform(row, member, model) {
  const values = [];
  const prep = member.prep;

  for (const feature of model.all_numeric_cols) {
    const raw = getRowValue(row, feature, model);
    const isMissing = raw === null;
    const value = isMissing ? prep.medians[feature] : raw;
    values.push((value - prep.means[feature]) / prep.stds[feature]);
    values.push(isMissing ? 1 : 0);
  }

  for (const col of model.categorical_cols) {
    const value = row[col] || "";
    for (const level of prep.levels[col]) {
      values.push(value === level ? 1 : 0);
    }
  }

  return values;
}

function scoreDave(row) {
  const scores = MODEL.ensemble.map((member) => {
    const x = transform(row, member, MODEL);
    const linear = x.reduce((sum, value, index) => sum + value * member.weights[index], member.intercept);
    return sigmoid(linear);
  });

  return scores.reduce((sum, value) => sum + value, 0) / scores.length;
}

function collectFormRow(form) {
  const data = new FormData(form);
  const row = {};

  for (const [key, value] of data.entries()) {
    row[key] = value;
  }

  for (const checkbox of form.querySelectorAll('input[type="checkbox"]')) {
    row[checkbox.name] = checkbox.checked ? 1 : 0;
    const paired60 = checkbox.name.replace("_36", "_60");
    if (paired60 !== checkbox.name) row[paired60] = checkbox.checked ? 1 : 0;
  }

  for (const input of form.querySelectorAll('input[type="number"]')) {
    const value = numberOrNull(input.value);
    if (value !== null) row[input.name] = value;
  }

  return row;
}

function riskCategory(probability) {
  if (probability >= 0.7) return "High";
  if (probability >= 0.35) return "Medium";
  return "Low";
}

function signalSummary(row) {
  const signals = [];
  const chronic = [
    ["cms_ccw_heart_failure_36", "Heart failure"],
    ["cms_ccw_chronic_kidney_disease_36", "CKD"],
    ["cms_ccw_diabetes_36", "Diabetes"],
    ["cms_ccw_hypertension_36", "Hypertension"],
    ["cms_ccw_ischemic_heart_disease_36", "Ischemic heart disease"],
    ["cms_ccw_liver_disease_cirrhosis_and_other_liver_conditions_36", "Liver disease"],
    ["cms_ccw_anemia_36", "Anemia"],
  ];
  const activeChronic = chronic.filter(([key]) => Number(row[key] || 0) === 1).map(([, label]) => label);
  if (activeChronic.length) signals.push(`Chronic flags: ${activeChronic.join(", ")}`);

  const officeVisits = numberOrNull(row.last_12months_officevisit_count);
  if (officeVisits !== null) signals.push(`${officeVisits} office visits in the last 12 months`);

  const specialties = [
    ["Specialty_internal_medicine", "internal medicine"],
    ["Specialty_nephrology", "nephrology"],
    ["Specialty_cardiology", "cardiology"],
  ].filter(([key]) => Number(row[key] || 0) === 1).map(([, label]) => label);
  if (specialties.length) signals.push(`Specialty engagement: ${specialties.join(", ")}`);

  const labs = ["latest_hemoglobin_24months", "latest_HbA1c_24months", "latest_AST_24months", "latest_ALT_24months"]
    .filter((key) => numberOrNull(row[key]) !== null)
    .map((key) => key.replace("latest_", "").replace("_24months", ""));
  if (labs.length) signals.push(`Labs entered: ${labs.join(", ")}`);

  return signals.length ? signals : ["No high-signal chronic, lab, or specialty inputs entered."];
}

const form = document.getElementById("predict-form");
const badge = document.getElementById("risk-badge");
const title = document.getElementById("result-title");
const copy = document.getElementById("result-copy");
const probability = document.getElementById("probability");
const predictedOutcome = document.getElementById("predicted-outcome");
const signalList = document.getElementById("signal-list");

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const row = collectFormRow(form);
  const score = scoreDave(row);
  const outcome = score >= MODEL.threshold ? 1 : 0;
  const category = riskCategory(score);

  badge.className = `risk-badge ${category.toLowerCase()}`;
  badge.textContent = `${category} risk`;
  probability.textContent = `${Math.round(score * 100)}%`;
  predictedOutcome.textContent = outcome === 1 ? "1 - likely return" : "0 - lower return risk";

  if (outcome === 1) {
    title.textContent = "Predictive model expects a future revisit";
    copy.textContent = "According to the DAVE model, this patient is likely to come back for ED or inpatient care. Consider triage review or referral workflow.";
  } else {
    title.textContent = "Predictive model does not expect a revisit";
    copy.textContent = "According to the DAVE model, this patient is less likely to come back for ED or inpatient care based on the entered parameters.";
  }

  signalList.innerHTML = "";
  for (const signal of signalSummary(row)) {
    const li = document.createElement("li");
    li.textContent = signal;
    signalList.appendChild(li);
  }
});
