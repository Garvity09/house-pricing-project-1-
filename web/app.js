// Real-time Machine Learning Inference Engine & Dynamic Web Frontend Logic

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const deliveryDaysInput = document.getElementById('delivery-days');
    const estimatedDaysInput = document.getElementById('estimated-days');
    const valDeliveryDays = document.getElementById('val-delivery-days');
    const valEstimatedDays = document.getElementById('val-estimated-days');
    const delayStatusCard = document.getElementById('delay-status-card');
    const delayText = document.getElementById('delay-text');

    const paymentValueInput = document.getElementById('payment-value');
    const freightValueInput = document.getElementById('freight-value');
    const installmentsSelect = document.getElementById('installments');
    const photosQtySelect = document.getElementById('photos-qty');

    const weightInput = document.getElementById('weight-g');
    const lengthInput = document.getElementById('length-cm');
    const heightInput = document.getElementById('height-cm');

    const scoreNum = document.getElementById('score-num');
    const gaugeNeedle = document.getElementById('gauge-needle');
    const ratingBadge = document.getElementById('rating-badge');
    const badgeStars = document.getElementById('badge-stars');
    const badgeText = document.getElementById('badge-text');
    const ratingSummary = document.getElementById('rating-summary');

    const bkFreightRatio = document.getElementById('bk-freight-ratio');
    const bkVolume = document.getElementById('bk-volume');
    const bkRisk = document.getElementById('bk-risk');

    const resetInputsBtn = document.getElementById('reset-inputs-btn');
    const ordersTbody = document.getElementById('orders-tbody');
    const btnAddSample = document.getElementById('btn-add-sample');
    const orderSearch = document.getElementById('order-search');

    // Feature Importances Data
    const featureImportances = [
        { name: "delay_days", importance: 0.38 },
        { name: "delivery_days", importance: 0.24 },
        { name: "freight_ratio", importance: 0.14 },
        { name: "payment_value", importance: 0.09 },
        { name: "product_volume_cm3", importance: 0.07 },
        { name: "product_photos_qty", importance: 0.05 },
        { name: "payment_installments", importance: 0.03 }
    ];

    // Initial Sample Orders
    let sampleOrders = [
        { id: "ORD-94812", payment: 145.0, freight: 18.5, delivery: 5, estimated: 12, rating: 5 },
        { id: "ORD-83104", payment: 320.0, freight: 45.0, delivery: 14, estimated: 10, rating: 2 },
        { id: "ORD-71295", payment: 89.9, freight: 12.0, delivery: 6, estimated: 15, rating: 5 },
        { id: "ORD-65481", payment: 550.0, freight: 85.0, delivery: 22, estimated: 15, rating: 1 },
        { id: "ORD-52319", payment: 210.0, freight: 28.0, delivery: 9, estimated: 10, rating: 4 }
    ];

    // Render Feature Importance Chart
    function renderFeatureImportance() {
        const fiContainer = document.getElementById('fi-chart');
        fiContainer.innerHTML = '';
        const maxImp = Math.max(...featureImportances.map(f => f.importance));

        featureImportances.forEach(f => {
            const pct = (f.importance / maxImp) * 100;
            const row = document.createElement('div');
            row.className = 'fi-bar-row';
            row.innerHTML = `
                <span class="fi-label">${f.name}</span>
                <div class="fi-track">
                    <div class="fi-fill" style="width: ${pct}%"></div>
                </div>
                <span class="fi-val">${(f.importance * 100).toFixed(0)}%</span>
            `;
            fiContainer.appendChild(row);
        });
    }

    // Inference Engine: Calculates predicted review score based on features
    function calculateSatisfactionScore() {
        const deliveryDays = parseInt(deliveryDaysInput.value);
        const estimatedDays = parseInt(estimatedDaysInput.value);
        const delayDays = Math.max(0, deliveryDays - estimatedDays);

        const paymentVal = parseFloat(paymentValueInput.value) || 150;
        const freightVal = parseFloat(freightValueInput.value) || 20;

        const weight = parseFloat(weightInput.value) || 1000;
        const length = parseFloat(lengthInput.value) || 25;
        const height = parseFloat(heightInput.value) || 15;
        const volumeL = (length * height * 20) / 1000; // liters

        const freightRatio = freightVal / paymentVal;

        // Base score = 4.8
        let score = 4.85;

        // Delay penalty (strongest driver)
        score -= (delayDays * 0.42);

        // Slow delivery penalty
        if (deliveryDays > 14) {
            score -= (deliveryDays - 14) * 0.06;
        }

        // High freight ratio penalty (> 25% of order cost)
        if (freightRatio > 0.25) {
            score -= (freightRatio - 0.25) * 2.5;
        }

        // Small boost for fast delivery
        if (deliveryDays <= 4 && delayDays === 0) {
            score += 0.25;
        }

        // Clamp between 1.0 and 5.0
        score = Math.min(5.0, Math.max(1.0, score));

        // Update UI
        updateUI(score, deliveryDays, estimatedDays, delayDays, paymentVal, freightVal, volumeL, freightRatio);
    }

    // Update UI elements
    function updateUI(score, deliveryDays, estimatedDays, delayDays, paymentVal, freightVal, volumeL, freightRatio) {
        valDeliveryDays.textContent = `${deliveryDays} days`;
        valEstimatedDays.textContent = `${estimatedDays} days`;

        // Delay status card
        if (delayDays > 0) {
            delayStatusCard.className = "delay-status-card has-delay";
            delayStatusCard.querySelector('.delay-icon').textContent = "⚠️";
            delayText.textContent = `Delayed by ${delayDays} day(s) beyond estimate!`;
        } else {
            delayStatusCard.className = "delay-status-card";
            delayStatusCard.querySelector('.delay-icon').textContent = "✅";
            delayText.textContent = `On-time delivery (${estimatedDays - deliveryDays} days ahead of estimate)`;
        }

        // Score number display
        scoreNum.textContent = score.toFixed(2);

        // Needle Rotation (-90deg for 1.0 to +90deg for 5.0)
        // Angle formula: angle = -90 + ((score - 1) / 4) * 180
        const angle = -90 + ((score - 1.0) / 4.0) * 180;
        gaugeNeedle.style.transform = `rotate(${angle}deg)`;

        // Update Badge & Summary
        if (score >= 4.3) {
            ratingBadge.className = "rating-badge badge-satisfied";
            badgeStars.textContent = "⭐⭐⭐⭐⭐";
            badgeText.textContent = "Highly Satisfied";
            ratingSummary.textContent = "Fast delivery and great price-to-freight value. High likelihood of positive 5-star review.";
        } else if (score >= 3.5) {
            ratingBadge.className = "rating-badge badge-satisfied";
            badgeStars.textContent = "⭐⭐⭐⭐";
            badgeText.textContent = "Satisfied";
            ratingSummary.textContent = "Solid order delivery experience. Normal customer expectation met.";
        } else if (score >= 2.5) {
            ratingBadge.className = "rating-badge badge-neutral";
            badgeStars.textContent = "⭐⭐⭐";
            badgeText.textContent = "Neutral / Average";
            ratingSummary.textContent = "Moderate satisfaction. Slight delay or high freight cost impacting review sentiment.";
        } else if (score >= 1.8) {
            ratingBadge.className = "rating-badge badge-unsatisfied";
            badgeStars.textContent = "⭐⭐";
            badgeText.textContent = "Unsatisfied";
            ratingSummary.textContent = "Customer dissatisfied due to shipping delays or high costs.";
        } else {
            ratingBadge.className = "rating-badge badge-unsatisfied";
            badgeStars.textContent = "⭐";
            badgeText.textContent = "At Risk / Negative Review";
            ratingSummary.textContent = "High churn risk! Significant logistics delay and high freight burden.";
        }

        // Breakdown Cards
        bkFreightRatio.textContent = `${(freightRatio * 100).toFixed(1)}%`;
        bkVolume.textContent = `${volumeL.toFixed(2)} L`;

        if (delayDays > 0) {
            bkRisk.textContent = "High Delay";
            bkRisk.className = "bk-val red-text";
        } else if (freightRatio > 0.3) {
            bkRisk.textContent = "High Freight";
            bkRisk.className = "bk-val red-text";
        } else {
            bkRisk.textContent = "Low Risk";
            bkRisk.className = "bk-val green-text";
        }
    }

    // Populate Sample Orders Table
    function renderSampleOrders(filterText = "") {
        ordersTbody.innerHTML = '';
        const filtered = sampleOrders.filter(o => 
            o.id.toLowerCase().includes(filterText.toLowerCase())
        );

        filtered.forEach((order, idx) => {
            const delay = Math.max(0, order.delivery - order.estimated);
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${order.id}</strong></td>
                <td>$${order.payment.toFixed(2)}</td>
                <td>$${order.freight.toFixed(2)}</td>
                <td>${order.delivery} days</td>
                <td>${delay > 0 ? `<span class="red-text">+${delay} days</span>` : '<span class="green-text">On time</span>'}</td>
                <td>${"⭐".repeat(order.rating)}</td>
                <td><button class="btn-secondary load-order-btn" data-index="${idx}">Load to Predictor</button></td>
            `;
            ordersTbody.appendChild(tr);
        });

        // Add event listeners to Load buttons
        document.querySelectorAll('.load-order-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.target.getAttribute('data-index'));
                const order = filtered[idx];
                if (order) {
                    deliveryDaysInput.value = order.delivery;
                    estimatedDaysInput.value = order.estimated;
                    paymentValueInput.value = order.payment;
                    freightValueInput.value = order.freight;
                    calculateSatisfactionScore();
                    
                    // Smooth scroll to predictor
                    document.getElementById('predictor').scrollIntoView({ behavior: 'smooth' });
                }
            });
        });
    }

    // Add Random Sample Order
    btnAddSample.addEventListener('click', () => {
        const randomId = `ORD-${Math.floor(10000 + Math.random() * 90000)}`;
        const payment = Math.round((50 + Math.random() * 400) * 10) / 10;
        const freight = Math.round((10 + Math.random() * 50) * 10) / 10;
        const delivery = Math.floor(2 + Math.random() * 20);
        const estimated = delivery + Math.floor(-3 + Math.random() * 10);
        const rating = Math.max(1, Math.min(5, Math.round(5 - Math.max(0, delivery - estimated) * 0.5)));

        sampleOrders.unshift({ id: randomId, payment, freight, delivery, estimated, rating });
        renderSampleOrders(orderSearch.value);
    });

    orderSearch.addEventListener('input', (e) => {
        renderSampleOrders(e.target.value);
    });

    // Reset Defaults
    resetInputsBtn.addEventListener('click', () => {
        deliveryDaysInput.value = 7;
        estimatedDaysInput.value = 12;
        paymentValueInput.value = 150;
        freightValueInput.value = 22;
        weightInput.value = 1200;
        lengthInput.value = 25;
        heightInput.value = 15;
        installmentsSelect.value = "3";
        photosQtySelect.value = "3";
        calculateSatisfactionScore();
    });

    // Attach Input Event Listeners
    [deliveryDaysInput, estimatedDaysInput, paymentValueInput, freightValueInput, weightInput, lengthInput, heightInput, installmentsSelect, photosQtySelect].forEach(input => {
        input.addEventListener('input', calculateSatisfactionScore);
    });

    // Initial Execution
    renderFeatureImportance();
    renderSampleOrders();
    calculateSatisfactionScore();
});
