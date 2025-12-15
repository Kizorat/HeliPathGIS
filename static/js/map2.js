// VERSIONE PULITA map2.js (senza palazzi.gpkg e debug)

// ============================================================
// --- Inizializzazione Mappa ---
// ============================================================
const map = L.map('map').setView([45.0, 9.0], 8);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

// ============================================================
// --- Layer principali ---
// ============================================================
const layerStrade = L.geoJSON(null, { style: { color: '#2b7bb9', weight: 2 } });

const heliIcon = L.divIcon({ html: '🚁', className: '', iconSize: [24, 24], iconAnchor: [12, 12] });
const layerAreeEli = L.geoJSON(null, {
    pointToLayer: (f, latlng) => L.marker(latlng, { icon: heliIcon }),
    onEachFeature: (f, layer) => layer.bindPopup(f.properties?.name || 'Area elicottero')
});

const layerNoFly = L.geoJSON(null, { style: { color: '#ff0000', weight: 2, fillOpacity: 0.15 } });
const layerGreenZone = L.geoJSON(null, { style: { color: '#27ae60', weight: 2, fillOpacity: 0.15 } });

const routeLayer = L.layerGroup().addTo(map);

// ============================================================
// --- Centroidi ---
// ============================================================
let centroidMarkers = [];
let selectedCentroid = null;

async function loadCentroids() {
    const resp = await fetch("/geojson?filename=Polygon_Centroid.gpkg");
    if (!resp.ok) return;
    
    const data = await resp.json();
    
    data.features.forEach(f => {
        const [lon, lat] = f.geometry.coordinates;
        const marker = L.circleMarker([lat, lon], { radius: 6, color: 'red', opacity: 0, fillOpacity: 0 });
        marker.feature_id = f.id ?? f.properties?.id;
        marker.on('click', () => selectCentroid(marker));
        marker.addTo(map);
        centroidMarkers.push(marker);
    });
}

function selectCentroid(marker) {
    if (selectedCentroid)
        selectedCentroid.setStyle({ color: 'red', radius: 6, opacity: 0, fillOpacity: 0 });

    selectedCentroid = marker;
    marker.setStyle({ color: 'orange', radius: 8, opacity: 1, fillOpacity: 0.8 });
}

// ============================================================
// --- Ospedali ---
// ============================================================
const defaultHospitalIcon = L.icon({
    iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png",
    shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
    iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
});

const selectedHospitalIcon = L.icon({
    iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-orange.png",
    shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
    iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
});

let selectedHospital = null;
let hospitalMarkers = [];

const layerOspedali = L.layerGroup().addTo(map);

async function attachHospitalClicks() {
    const resp = await fetch("/geojson?filename=Centroidi Ospedali.gpkg");
    if (!resp.ok) return;

    const data = await resp.json();

    data.features.forEach(f => {
        const [lon, lat] = f.geometry.coordinates;
        const marker = L.marker([lat, lon], { icon: defaultHospitalIcon })
            .on("click", () => {
                if (selectedHospital) selectedHospital.setIcon(defaultHospitalIcon);
                selectedHospital = marker;
                marker.setIcon(selectedHospitalIcon);
            })
            .bindPopup(f.properties?.name || "Ospedale");

        marker.addTo(layerOspedali);
        hospitalMarkers.push(marker);
    });
}

// ============================================================
// --- Caricamento Layer Standard ---
// ============================================================
async function loadLayer(layer, filename) {
    try {
        const resp = await fetch(`/geojson?filename=${encodeURIComponent(filename)}`);
        if (!resp.ok) return;
        
        const data = await resp.json();
        layer.addData(data).addTo(map);
    } catch (err) {
        // Silenzioso in caso di errore
    }
}

// Carica i layer
loadLayer(layerStrade, "Strade_Lombardia.geojson");
loadLayer(layerAreeEli, "Area atterraggio elicottero.gpkg");
loadLayer(layerNoFly, "Lombardia_no_fly.gpkg");
loadLayer(layerGreenZone, "fly_zone_Lombardia.gpkg");

loadCentroids();
attachHospitalClicks();

// ============================================================
// --- Checkbox Toggle ---
// ============================================================
document.getElementById('chkStrade').addEventListener('change', function () {
    this.checked ? layerStrade.addTo(map) : map.removeLayer(layerStrade);
});

document.getElementById('chkAreeEli').addEventListener('change', function () {
    this.checked ? layerAreeEli.addTo(map) : map.removeLayer(layerAreeEli);
});

document.getElementById('chkAreeOsp').addEventListener('change', function () {
    this.checked ? map.addLayer(layerOspedali) : map.removeLayer(layerOspedali);
});

// ============================================================
// --- Percorso Auto ---
// ============================================================
async function callAutoPath() {
    if (!selectedCentroid) return alert("Seleziona prima un centroide!");
    if (!selectedHospital) return alert("Seleziona prima un ospedale!");

    const centroidLatLng = selectedCentroid.getLatLng();
    const hospitalLatLng = selectedHospital.getLatLng();

    const url = `/auto-path?start_lat=${centroidLatLng.lat}&start_lon=${centroidLatLng.lng}&end_lat=${hospitalLatLng.lat}&end_lon=${hospitalLatLng.lng}`;
    const resp = await fetch(url);

    if (!resp.ok) return alert("Errore server");

    const data = await resp.json();
    routeLayer.clearLayers();

    const feature = data.features[0];
    if (feature.geometry.type !== "MultiLineString") {
        alert("Percorso non valido dal server");
        return;
    }

    const segments = feature.geometry.coordinates;
    let allLatLngs = [];

    segments.forEach(seg => {
        const segLatLng = seg.map(c => [c[1], c[0]]);
        allLatLngs.push(segLatLng);
        L.polyline(segLatLng, { color: "yellow", weight: 5 }).addTo(routeLayer);
    });

    const flat = allLatLngs.flat();
    if (flat.length < 2) return alert("Percorso troppo corto o non valido");

    map.fitBounds(flat, { padding: [40, 40] });

    L.popup({ maxWidth: 300 })
        .setLatLng(flat[0])
        .setContent(
            `Percorso auto: ${feature.properties.distance_km} km<br>` +
            `Tempo stimato: ${feature.properties.time_hhmm}`
        )
        .openOn(map);
}

// ============================================================
// --- Percorso Elicottero ---
// ============================================================
async function callHeliPath() {
    if (!selectedCentroid) return alert("Seleziona prima un centroide!");
    if (!selectedHospital) return alert("Seleziona prima un ospedale!");

    const centroidLatLng = selectedCentroid.getLatLng();
    const hospitalLatLng = selectedHospital.getLatLng();

    const url = `/heli-path?start_lat=${centroidLatLng.lat}&start_lon=${centroidLatLng.lng}&end_lat=${hospitalLatLng.lat}&end_lon=${hospitalLatLng.lng}`;
    const resp = await fetch(url);

    if (!resp.ok) {
        const err = await resp.json();
        alert("Errore server: " + (err.error || resp.statusText));
        return;
    }

    const data = await resp.json();
    routeLayer.clearLayers();

    const routes = data.features.filter(f => f.geometry && f.geometry.type === "MultiLineString");

    routes.forEach(routeFeature => {
        const mode = routeFeature.properties.mode || "";
        const color = mode === "heli" ? "yellow" : "red";

        routeFeature.geometry.coordinates.forEach(seg => {
            const segLatLng = seg.map(c => [c[1], c[0]]);
            const poly = L.polyline(segLatLng, { color, weight: 5 }).addTo(routeLayer);

            poly.bindPopup(
                `${mode === "heli" ? "Percorso Eli" : "Percorso Auto"}<br>` +
                `Distanza: ${routeFeature.properties.distance_km} km<br>` +
                `Tempo stimato: ${routeFeature.properties.time_hhmm}`
            );
        });
    });

    const helipad = data.features.find(f => f.properties.type === "helipad");
    if (helipad) {
        const [lon, lat] = helipad.geometry.coordinates;
        L.marker([lat, lon], { icon: heliIcon }).addTo(routeLayer);
    }

    const hospital = data.features.find(f => f.properties.type === "hospital");
    if (hospital) {
        const [lon, lat] = hospital.geometry.coordinates;
        L.marker([lat, lon], { icon: selectedHospitalIcon }).addTo(routeLayer);
    }

    const totals = data.features.find(f => f.geometry === null && f.properties.distance_km_total !== undefined);
    if (totals) {
        L.popup({ maxWidth: 300 })
            .setLatLng(centroidLatLng)
            .setContent(
                `Percorso totale:<br>` +
                `Distanza: ${totals.properties.distance_km_total} km<br>` +
                `Tempo stimato: ${totals.properties.time_hhmm_total}`
            )
            .openOn(map);
    }

    const flat = routes.flatMap(f => f.geometry.coordinates.flat().map(c => [c[1], c[0]]));
    if (flat.length >= 2) map.fitBounds(flat, { padding: [40, 40] });
}

// ============================================================
// --- Percorso Migliore (Auto + Elicottero) ---
// ============================================================
async function callAllPath() {
    if (!selectedCentroid) return alert("Seleziona prima un centroide!");
    if (!selectedHospital) return alert("Seleziona prima un ospedale!");

    const centroidLatLng = selectedCentroid.getLatLng();
    const hospitalLatLng = selectedHospital.getLatLng();

    const url = `/best-path?start_lat=${centroidLatLng.lat}&start_lon=${centroidLatLng.lng}&end_lat=${hospitalLatLng.lat}&end_lon=${hospitalLatLng.lng}`;
    const resp = await fetch(url);

    if (!resp.ok) {
        const err = await resp.json().catch(() => null);
        alert("Errore server: " + (err?.error || resp.statusText));
        return;
    }

    const data = await resp.json();
    routeLayer.clearLayers();

    const routeFeatures = data.features.filter(f => f.geometry && f.geometry.type === "MultiLineString");

    let allPoints = [];

    routeFeatures.forEach(feature => {
        const mode = feature.properties.mode;
        const color = mode === "heli" ? "yellow" : (mode === "auto" ? "grey" : "red");

        feature.geometry.coordinates.forEach(seg => {
            const segLatLng = seg.map(c => [c[1], c[0]]);
            allPoints.push(...segLatLng);

            const poly = L.polyline(segLatLng, { color, weight: 5 }).addTo(routeLayer);

            poly.bindPopup(
                `${mode === "heli" ? "Tratto Elicottero" : "Tratto Auto"}<br>` +
                `Distanza: ${feature.properties.distance_km} km<br>` +
                `Tempo: ${feature.properties.time_hhmm}`
            );
        });
    });

    const helipad = data.features.find(f => f.properties.type === "helipad");
    if (helipad) {
        const [lon, lat] = helipad.geometry.coordinates;
        L.marker([lat, lon], { icon: heliIcon }).addTo(routeLayer);
    }

    const hospital = data.features.find(f => f.properties.type === "hospital");
    if (hospital) {
        const [lon, lat] = hospital.geometry.coordinates;
        L.marker([lat, lon], { icon: selectedHospitalIcon }).addTo(routeLayer);
    }

    const summary = data.features.find(f => f.properties.is_summary === true);
    if (summary) {
        const props = summary.properties;
        
        let comparisonHTML = `<div style="font-size: 14px; max-width: 350px;">`;
        comparisonHTML += `<h3 style="color: #2c3e50; margin-bottom: 10px; border-bottom: 2px solid #3498db; padding-bottom: 5px;">🏆 CONFRONTO PERCORSI</h3>`;
        
        if (props.comparison_message) {
            comparisonHTML += `<div style="white-space: pre-line; line-height: 1.4;">${props.comparison_message}</div>`;
        } else {
            comparisonHTML += `<div><strong>Miglior percorso:</strong> ${props.best_mode.toUpperCase()}</div>`;
            
            if (props.auto_available) {
                comparisonHTML += `<div style="margin-top: 8px; padding: 5px; background: ${props.best_mode === 'auto' ? '#d4edda' : '#f8f9fa'}; border-left: 4px solid ${props.best_mode === 'auto' ? '#28a745' : '#6c757d'};">`;
                comparisonHTML += `<strong>🚗 Auto:</strong> ${props.auto_km || '?'} km, ${props.auto_time_min || '?'} min`;
                comparisonHTML += `</div>`;
            }
            
            if (props.heli_available) {
                comparisonHTML += `<div style="margin-top: 5px; padding: 5px; background: ${props.best_mode === 'heli' ? '#d4edda' : '#f8f9fa'}; border-left: 4px solid ${props.best_mode === 'heli' ? '#ffc107' : '#6c757d'};">`;
                comparisonHTML += `<strong>🚁 Elicottero:</strong> ${props.heli_km || '?'} km, ${props.heli_time_min || '?'} min`;
                comparisonHTML += `</div>`;
            }
        }
        
        comparisonHTML += `<div style="margin-top: 15px; font-size: 12px; color: #666; border-top: 1px solid #eee; padding-top: 10px;">`;
        comparisonHTML += `Clicca su un percorso per i dettagli`;
        comparisonHTML += `</div>`;
        comparisonHTML += `</div>`;
        
        L.popup({ maxWidth: 400, autoClose: false, closeOnClick: false })
            .setLatLng([(centroidLatLng.lat + hospitalLatLng.lat) / 2, 
                       (centroidLatLng.lng + hospitalLatLng.lng) / 2])
            .setContent(comparisonHTML)
            .openOn(map);
    }

    if (allPoints.length >= 2) map.fitBounds(allPoints, { padding: [40, 40] });
}

// ============================================================
// --- Event Listener Bottoni ---
// ============================================================
document.getElementById("autoPathBtn").addEventListener("click", callAutoPath);
document.getElementById("heliPathBtn").addEventListener("click", callHeliPath);
document.getElementById("clearRoutes").addEventListener("click", () => routeLayer.clearLayers());
document.getElementById("specialBtn").addEventListener("click", callAllPath);