// VERSIONE COMPLETA map2.js (Colori aggiornati: Grigio, Rosso, Giallo)

// ============================================================
// --- Inizializzazione Mappa ---
// ============================================================
const map = L.map('map').setView([45.0, 9.0], 8);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

// ============================================================
// --- Helper: Mostra messaggi utente come Popup ---
// ============================================================
function showUserMessage(message) {
    // Apre un popup al centro della vista attuale della mappa
    L.popup()
        .setLatLng(map.getCenter())
        .setContent(`<div style="text-align:center; padding: 10px;"><strong>⚠️ Attenzione</strong><br>${message}</div>`)
        .openOn(map);
}

// ============================================================
// --- Contatori globali ---
// ============================================================
let totalHospitals = 0;
let totalHelipads = 0;

// ============================================================
// --- Layer principali ---
// ============================================================
const layerStrade = L.geoJSON(null, { style: { color: '#2b7bb9', weight: 2 } });

const heliIcon = L.divIcon({ html: '🚁', className: '', iconSize: [24, 24], iconAnchor: [12, 12] });
const layerAreeEli = L.geoJSON(null, {
    pointToLayer: (f, latlng) => L.marker(latlng, { icon: heliIcon }),
    onEachFeature: (f, layer) => layer.bindPopup(f.properties?.DENOMINAZI || f.properties?.denominazi || f.properties?.name || 'Area elicottero')
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
    
    // Filtra solo gli ospedali che hanno un nome valido
    const validHospitals = data.features.filter(f => {
        const name = f.properties?.name;
        return name && name !== "" && name.toLowerCase() !== "none" && name.trim() !== "";
    });
    
    totalHospitals = validHospitals.length;
    
    console.log(`📊 Caricati ${totalHospitals} ospedali (con nome valido)`);

    validHospitals.forEach(f => {
        const [lon, lat] = f.geometry.coordinates;
        const hospitalName = f.properties.name;
        
        const marker = L.marker([lat, lon], { icon: defaultHospitalIcon })
            .on("click", () => {
                if (selectedHospital) selectedHospital.setIcon(defaultHospitalIcon);
                selectedHospital = marker;
                marker.setIcon(selectedHospitalIcon);
            })
            .bindPopup(`<strong>🏥 ${hospitalName}</strong>`);

        marker.addTo(layerOspedali);
        hospitalMarkers.push(marker);
    });
    
    updateStatsDisplay();
}

// ============================================================
// --- Caricamento Layer Standard ---
// ============================================================
async function loadLayer(layer, filename) {
    try {
        const resp = await fetch(`/geojson?filename=${encodeURIComponent(filename)}`);
        if (!resp.ok) return;
        
        const data = await resp.json();
        
        // Conta eliporti
        if (filename === "Area atterraggio elicottero.gpkg") {
            totalHelipads = data.features.length;
            console.log(`📊 Caricati ${totalHelipads} eliporti`);
            updateStatsDisplay();
        }
        
        layer.addData(data).addTo(map);
    } catch (err) {
        // Silenzioso in caso di errore
    }
}

// ============================================================
// --- Funzione per aggiornare il display delle statistiche ---
// ============================================================
function updateStatsDisplay() {
    if (totalHospitals > 0 || totalHelipads > 0) {
        console.log(`📊 STATISTICHE: Ospedali: ${totalHospitals}, Eliporti: ${totalHelipads}`);
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
// --- Funzioni Helper per Spinner ---
// ============================================================
function showSpinner(message = "Calcolo percorso in corso...") {
    const loadingText = document.querySelector('#loadingSpinner p');
    if (loadingText) {
        loadingText.textContent = message;
    }
    document.getElementById('loadingSpinner').style.display = 'flex';
    document.getElementById('map').classList.add('map-loading');
}

function hideSpinner() {
    document.getElementById('loadingSpinner').style.display = 'none';
    document.getElementById('map').classList.remove('map-loading');
}

// ============================================================
// --- Percorso Auto ---
// ============================================================
async function callAutoPath() {
    if (!selectedCentroid) return showUserMessage("Seleziona prima un centroide!");
    if (!selectedHospital) return showUserMessage("Seleziona prima un ospedale!");

    showSpinner("Calcolo percorso auto...");

    try {
        const centroidLatLng = selectedCentroid.getLatLng();
        const hospitalLatLng = selectedHospital.getLatLng();

        const url = `/auto-path?start_lat=${centroidLatLng.lat}&start_lon=${centroidLatLng.lng}&end_lat=${hospitalLatLng.lat}&end_lon=${hospitalLatLng.lng}`;
        const resp = await fetch(url);

        if (!resp.ok) {
            hideSpinner();
            return showUserMessage("Errore server durante il calcolo");
        }

        const data = await resp.json();
        routeLayer.clearLayers();

        const feature = data.features[0];
        if (feature.geometry.type !== "MultiLineString") {
            hideSpinner();
            showUserMessage("Percorso non valido ricevuto dal server");
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
        if (flat.length < 2) {
            hideSpinner();
            return showUserMessage("Percorso troppo corto o non valido");
        }

        map.fitBounds(flat, { padding: [40, 40] });

        L.popup({ maxWidth: 300 })
            .setLatLng(flat[0])
            .setContent(
                `Percorso auto: ${feature.properties.distance_km} km<br>` +
                `Tempo stimato: ${feature.properties.time_hhmm}`
            )
            .openOn(map);

    } catch (error) {
        console.error("Errore in callAutoPath:", error);
        showUserMessage("Errore imprevisto durante il calcolo del percorso");
    } finally {
        hideSpinner();
    }
}

// ============================================================
// --- Percorso Elicottero ---
// ============================================================
async function callHeliPath() {
    if (!selectedCentroid) return showUserMessage("Seleziona prima un centroide!");
    if (!selectedHospital) return showUserMessage("Seleziona prima un ospedale!");

    showSpinner("Calcolo percorso elicottero...");

    try {
        const centroidLatLng = selectedCentroid.getLatLng();
        const hospitalLatLng = selectedHospital.getLatLng();

        const url = `/heli-path?start_lat=${centroidLatLng.lat}&start_lon=${centroidLatLng.lng}&end_lat=${hospitalLatLng.lat}&end_lon=${hospitalLatLng.lng}`;
        const resp = await fetch(url);

        if (!resp.ok) {
            const err = await resp.json();
            hideSpinner();
            showUserMessage("Errore server: " + (err.error || resp.statusText));
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

    } catch (error) {
        console.error("Errore in callHeliPath:", error);
        showUserMessage("Errore durante il calcolo del percorso elicottero");
    } finally {
        hideSpinner();
    }
}

// ============================================================
// --- Percorso Migliore ---
// ============================================================
async function callAllPath() {
    if (!selectedCentroid) return showUserMessage("Seleziona prima un centroide!");
    if (!selectedHospital) return showUserMessage("Seleziona prima un ospedale!");

    showSpinner("Calcolo percorso ottimale...");

    try {
        const centroidLatLng = selectedCentroid.getLatLng();
        const hospitalLatLng = selectedHospital.getLatLng();

        const url = `/best-path?start_lat=${centroidLatLng.lat}&start_lon=${centroidLatLng.lng}&end_lat=${hospitalLatLng.lat}&end_lon=${hospitalLatLng.lng}`;
        const resp = await fetch(url);

        if (!resp.ok) {
            const err = await resp.json().catch(() => null);
            hideSpinner();
            showUserMessage("Errore server: " + (err?.error || resp.statusText));
            return;
        }

        const data = await resp.json();
        routeLayer.clearLayers();

        const routeFeatures = data.features.filter(f => f.geometry && f.geometry.type === "MultiLineString");

        let allPoints = [];

        routeFeatures.forEach(feature => {
            const mode = feature.properties.mode;
            const color = mode === "heli" ? "yellow" : (mode === "auto" ? "red" : "gray");

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

    } catch (error) {
        console.error("Errore in callAllPath:", error);
        showUserMessage("Errore durante il calcolo del percorso ottimale");
    } finally {
        hideSpinner();
    }
}


// ============================================================
// --- Percorso Ospedale più Vicino
// ============================================================
async function callNearestHospitalViaHeliport() {
    if (!selectedCentroid) return showUserMessage("Seleziona prima un centroide!");

    showSpinner("Calcolo percorso ottimale...");

    try {
        const centroidLatLng = selectedCentroid.getLatLng();
        const url = `/emergency-route?start_lat=${centroidLatLng.lat}&start_lon=${centroidLatLng.lng}`;
        const resp = await fetch(url);

        if (!resp.ok) {
            const err = await resp.json().catch(() => null);
            hideSpinner();
            showUserMessage("Errore server: " + (err?.error || resp.statusText));
            return;
        }

        const data = await resp.json();
        console.log("Dati ricevuti:", data); // DEBUG
        routeLayer.clearLayers();

        // Conta le feature per debug
        const routeFeatures = data.features.filter(f => f.geometry && f.geometry.type === "MultiLineString");
        console.log(`${routeFeatures.length} tratti percorsi trovati`);
        routeFeatures.forEach((f, i) => {
            console.log(`Tratto ${i}: mode=${f.properties.mode}, distance=${f.properties.distance_km}`);
        });

        const hospitalFeature = data.features.find(f => f.properties.type === "hospital");
        const helipadFeature = data.features.find(f => f.properties.type === "helipad");
        const summaryFeature = data.features.find(f => f.properties.is_summary === true);

        let allPoints = [];

        // --- DISEGNO PERCORSI CON STILE SIMILE A callAllPath ---
        routeFeatures.forEach(feature => {
            const mode = feature.properties.mode;
            console.log(`Disegnando tratto con mode: ${mode}`); // DEBUG
            
            // Usa gli stessi colori di callAllPath per consistenza
            let color, weight = 5, opacity = 0.8, popupPrefix;
            
            if (mode === "heli_flight") {
                color = "yellow";
                popupPrefix = "🚁 Tratto Elicottero";
                weight = 6;
                opacity = 1;
            } else if (mode === "auto_to_helipad") {
                color = "red";
                popupPrefix = "🚗 Auto verso Eliporto";
                weight = 5;
                opacity = 0.9;
            } else if (mode === "direct_auto") {
                color = "gray";
                popupPrefix = "🚗 Auto diretto (confronto)";
                weight = 4;
                opacity = 0.6;
            } else {
                color = "blue";
                popupPrefix = "Tratto percorso";
            }

            // Verifica che ci siano coordinate
            if (!feature.geometry || !feature.geometry.coordinates) {
                console.warn("Feature senza coordinate:", feature);
                return;
            }

            feature.geometry.coordinates.forEach((seg, segIndex) => {
                if (!seg || seg.length === 0) {
                    console.warn(`Segmento ${segIndex} vuoto`);
                    return;
                }
                
                const segLatLng = seg.map(c => {
                    // Gestisce sia [lon, lat] che [lon, lat, alt]
                    if (Array.isArray(c) && c.length >= 2) {
                        return [c[1], c[0]]; // Converti [lon, lat] a [lat, lon]
                    }
                    console.warn("Coordinate non valide:", c);
                    return [0, 0];
                }).filter(coord => coord[0] !== 0 && coord[1] !== 0);
                
                if (segLatLng.length === 0) {
                    console.warn("Nessuna coordinata valida nel segmento");
                    return;
                }
                
                allPoints.push(...segLatLng);

                const poly = L.polyline(segLatLng, { 
                    color, 
                    weight: weight,
                    opacity: opacity,
                    lineJoin: 'round',
                    lineCap: 'round'
                }).addTo(routeLayer);

                // Popup uniformato a callAllPath
                let popupContent = `<strong>${popupPrefix}</strong><br>`;
                if (feature.properties.distance_km) {
                    popupContent += `📏 Distanza: ${feature.properties.distance_km} km<br>`;
                }
                if (feature.properties.time_hhmm) {
                    popupContent += `⏱️ Tempo: ${feature.properties.time_hhmm}`;
                } else if (feature.properties.time_min) {
                    popupContent += `⏱️ Tempo: ${feature.properties.time_min} min`;
                }
                
                // Aggiungi note specifiche se presenti
                if (feature.properties.is_best === true) {
                    popupContent += `<br><span style="color: green; font-weight: bold;">✓ Percorso consigliato</span>`;
                }
                if (feature.properties.description) {
                    popupContent += `<br><small>${feature.properties.description}</small>`;
                }
                
                poly.bindPopup(popupContent);
            });
        });

        console.log(`Totale punti disegnati: ${allPoints.length}`);

        // --- MARKER UNIFORMATI A callAllPath ---
        // Marker eliporto (sostituisce l'elicottero di callAllPath)
        if (helipadFeature && helipadFeature.geometry && helipadFeature.geometry.coordinates) {
            const [lon, lat] = helipadFeature.geometry.coordinates;
            console.log(`Aggiungendo marker eliporto a ${lat}, ${lon}`);
            const marker = L.marker([lat, lon], { icon: heliIcon }).addTo(routeLayer);
            
            const helipadDescription = helipadFeature.properties.description || "🚁 Eliporto di transizione";
            marker.bindPopup(`<strong>${helipadDescription}</strong>`);
            allPoints.push([lat, lon]);
        } else {
            console.warn("Nessun eliporto trovato o coordinate mancanti");
        }

        // Marker ospedale
        if (hospitalFeature && hospitalFeature.geometry && hospitalFeature.geometry.coordinates) {
            const [lon, lat] = hospitalFeature.geometry.coordinates;
            console.log(`Aggiungendo marker ospedale a ${lat}, ${lon}`);
            const marker = L.marker([lat, lon], { icon: selectedHospitalIcon }).addTo(routeLayer);
            
            const hospitalName = hospitalFeature.properties.name || "Ospedale";
            const hospitalDescription = hospitalFeature.properties.description || `🏥 ${hospitalName}`;
            marker.bindPopup(`<strong>${hospitalDescription}</strong>`);
            allPoints.push([lat, lon]);
        } else {
            console.warn("Nessun ospedale trovato o coordinate mancanti");
        }

        // --- POPUP RIEPILOGO UNIFORMATO A callAllPath ---
        if (summaryFeature) {
            console.log("Creando popup riepilogo");
            const props = summaryFeature.properties;
            const hospitalLatLng = hospitalFeature && hospitalFeature.geometry ? 
                [hospitalFeature.geometry.coordinates[1], hospitalFeature.geometry.coordinates[0]] : 
                null;
            
            // Calcola punto medio per posizionare il popup (come in callAllPath)
            const popupLatLng = hospitalLatLng ? 
                [(centroidLatLng.lat + hospitalLatLng[0]) / 2, 
                 (centroidLatLng.lng + hospitalLatLng[1]) / 2] :
                centroidLatLng;
            
            let summaryHTML = `<div style="font-size: 14px; max-width: 400px;">`;
            summaryHTML += `<h3 style="color: #2c3e50; margin-bottom: 10px; border-bottom: 2px solid #3498db; padding-bottom: 5px;">🏆 PERCORSO EMERGENZA</h3>`;
            
            // Mostra messaggio di confronto se disponibile (il backend restituisce SEMPRE summary_message)
            if (props.summary_message) {
                summaryHTML += `<div style="white-space: pre-line; line-height: 1.4; margin-bottom: 10px;">`;
                summaryHTML += `${props.summary_message}`;
                summaryHTML += `</div>`;
            }
            
            // Dettagli tecnici aggiuntivi
            summaryHTML += `<div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">`;
            summaryHTML += `<strong>📊 Dettagli tecnici:</strong><br>`;
            
            // Percorso auto diretto (SEMPRE disponibile secondo il backend)
            if (props.direct_auto_distance_km !== undefined && props.direct_auto_time_min !== undefined) {
                summaryHTML += `<div style="margin-top: 5px; padding: 5px; background: ${props.best_mode === 'auto' ? '#d4edda' : '#ffffff'}; border-left: 4px solid gray;">`;
                summaryHTML += `<strong>🚗 Solo auto:</strong> ${props.direct_auto_distance_km} km, ${props.direct_auto_time_min} min`;
                if (props.best_mode === 'auto') {
                    summaryHTML += ` <span style="color: green; font-weight: bold;">✓ CONSIGLIATO</span>`;
                }
                summaryHTML += `</div>`;
            }
            
            // Percorso auto+elicottero (disponibile solo se heli_available è true)
            if (props.heli_available && props.heli_total_distance_km !== undefined && props.heli_total_time_min !== undefined) {
                summaryHTML += `<div style="margin-top: 5px; padding: 5px; background: ${props.best_mode === 'heli' ? '#d4edda' : '#ffffff'}; border-left: 4px solid #ffc107;">`;
                summaryHTML += `<strong>🚗+🚁 Auto+Elicottero:</strong> ${props.heli_total_distance_km} km, ${props.heli_total_time_min} min`;
                if (props.best_mode === 'heli') {
                    summaryHTML += ` <span style="color: green; font-weight: bold;">✓ CONSIGLIATO</span>`;
                }
                summaryHTML += `</div>`;
            } else if (!props.heli_available) {
                summaryHTML += `<div style="margin-top: 5px; padding: 5px; background: #f8d7da; border-left: 4px solid #dc3545;">`;
                summaryHTML += `<strong>🚁 Elicottero:</strong> Non disponibile`;
                summaryHTML += `</div>`;
            }
            
            summaryHTML += `</div>`;
            
            // Tempo di calcolo (se disponibile)
            if (props.computation_time !== undefined) {
                summaryHTML += `<div style="margin-top: 10px; font-size: 11px; color: #999; text-align: right;">`;
                summaryHTML += `⚡ Calcolato in ${props.computation_time}s`;
                summaryHTML += `</div>`;
            }
            
            // Legenda percorsi
            summaryHTML += `<div style="margin-top: 15px; font-size: 12px; color: #666; border-top: 1px solid #eee; padding-top: 10px;">`;
            summaryHTML += `<strong>📍 Legenda percorsi:</strong><br>`;
            summaryHTML += `<span style="color: red; font-weight: bold;">━━━</span> Auto verso eliporto<br>`;
            summaryHTML += `<span style="color: yellow; font-weight: bold; text-shadow: 0px 0px 2px #000;">━━━</span> Volo in elicottero<br>`;
            summaryHTML += `<span style="color: gray; font-weight: bold;">━━━</span> Auto diretto (confronto)<br>`;
            summaryHTML += `<small style="color: #999;">Clicca su un percorso per i dettagli</small>`;
            summaryHTML += `</div>`;
            
            summaryHTML += `</div>`;
            
            // Crea popup (stesse opzioni di callAllPath)
            L.popup({ 
                maxWidth: 450, 
                autoClose: false, 
                closeOnClick: false,
                className: 'emergency-summary-popup'
            })
                .setLatLng(popupLatLng)
                .setContent(summaryHTML)
                .openOn(map);
        } else {
            console.warn("Nessun riepilogo trovato nei dati");
        }

        // Fit bounds (stesso comportamento di callAllPath)
        if (allPoints.length >= 2) {
            console.log(`Fitting bounds su ${allPoints.length} punti`);
            map.fitBounds(allPoints, { padding: [50, 50] });
        } else {
            console.warn("Punti insufficienti per fitBounds");
            // Centra sulla posizione iniziale
            map.setView(centroidLatLng, 12);
        }

    } catch (error) {
        console.error("Errore in callNearestHospitalViaHeliport:", error);
        showUserMessage("Errore durante il calcolo del percorso di emergenza");
    } finally {
        hideSpinner();
    }
}


document.getElementById("autoPathBtn").addEventListener("click", callAutoPath);
document.getElementById("heliPathBtn").addEventListener("click", callHeliPath);
document.getElementById("clearRoutes").addEventListener("click", () => routeLayer.clearLayers());
document.getElementById("specialBtn").addEventListener("click", callAllPath);
document.getElementById("nearestHospitalViaHeliportBtn")?.addEventListener("click", callNearestHospitalViaHeliport);