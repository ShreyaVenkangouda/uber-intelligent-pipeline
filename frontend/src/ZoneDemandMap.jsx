import { useEffect, useMemo, useState } from "react";
import {
    GeoJSON,
    MapContainer,
    TileLayer,
    useMap,
} from "react-leaflet";
import L from "leaflet";

const GEOJSON_URL = "/data/taxi_zones.geojson";

function getZoneId(properties) {
    const candidates = [
        "locationid",
        "LocationID",
        "location_id",
        "OBJECTID",
        "objectid",
        "zone_id",
    ];

    for (const key of candidates) {
        const value = properties?.[key];

        if (value !== undefined && value !== null && value !== "") {
            return String(value);
        }
    }

    return null;
}

function FitZoneBounds({ data }) {
    const map = useMap();

    useEffect(() => {
        if (!data?.features?.length) return;

        const layer = L.geoJSON(data);
        const bounds = layer.getBounds();

        if (bounds.isValid()) {
            map.fitBounds(bounds, {
                padding: [20, 20],
                maxZoom: 12,
            });
        }
    }, [data, map]);

    return null;
}

function getColor(value, maximum) {
    if (value == null || !Number.isFinite(value)) {
        return "#334155";
    }

    const ratio = maximum > 0 ? value / maximum : 0;

    if (ratio >= 0.8) return "#10b981";
    if (ratio >= 0.6) return "#34d399";
    if (ratio >= 0.4) return "#65a30d";
    if (ratio >= 0.2) return "#ca8a04";

    return "#64748b";
}

export default function ZoneDemandMap({ predictions = [] }) {
    const [geoData, setGeoData] = useState(null);
    const [error, setError] = useState("");
    const [selectedZone, setSelectedZone] = useState(null);

    // Load the NYC taxi-zone GeoJSON.
    useEffect(() => {
        const controller = new AbortController();

        async function loadZones() {
            try {
                setError("");

                const response = await fetch(GEOJSON_URL, {
                    signal: controller.signal,
                });

                if (!response.ok) {
                    throw new Error(
                        `Could not load taxi zones (HTTP ${response.status}).`,
                    );
                }

                const data = await response.json();

                if (!data.features || !Array.isArray(data.features)) {
                    throw new Error("The GeoJSON file has no features array.");
                }

                setGeoData(data);
            } catch (err) {
                if (err.name !== "AbortError") {
                    setError(err.message || "Could not load taxi-zone boundaries.");
                }
            }
        }

        loadZones();

        return () => controller.abort();
    }, []);

    // Match predictions to GeoJSON location IDs.
    const demandByZone = useMemo(() => {
        const result = new Map();

        for (const prediction of predictions) {
            if (prediction.pickup_location == null) continue;

            const zoneId = String(prediction.pickup_location);
            const demand = Number(prediction.predicted_demand);

            if (Number.isFinite(demand)) {
                result.set(zoneId, demand);
            }
        }

        return result;
    }, [predictions]);

    const maximumDemand = useMemo(() => {
        const values = [...demandByZone.values()];

        return values.length > 0 ? Math.max(...values) : 0;
    }, [demandByZone]);

    const matchedZoneCount = useMemo(() => {
        if (!geoData?.features) return 0;

        return geoData.features.filter((feature) => {
            const zoneId = getZoneId(feature.properties);
            return zoneId !== null && demandByZone.has(zoneId);
        }).length;
    }, [geoData, demandByZone]);

    function styleFeature(feature) {
        const zoneId = getZoneId(feature.properties);
        const demand = zoneId === null
            ? undefined
            : demandByZone.get(zoneId);

        return {
            fillColor: getColor(demand, maximumDemand),
            weight: 1,
            opacity: 0.9,
            color: "#94a3b8",
            fillOpacity: demand == null ? 0.18 : 0.8,
        };
    }

    function onEachFeature(feature, layer) {
        const properties = feature.properties ?? {};
        const zoneId = getZoneId(properties);
        const demand = zoneId === null
            ? undefined
            : demandByZone.get(zoneId);

        const zoneName =
            properties.zone ??
            properties.Zone ??
            properties.name ??
            `Zone ${zoneId ?? "unknown"}`;

        const borough =
            properties.borough ??
            properties.Borough ??
            "Unknown";

        const tooltipText =
            demand == null
                ? `${zoneName} — no prediction`
                : `${zoneName} — ${demand.toFixed(3)} predicted trips`;

        layer.bindTooltip(tooltipText);

        layer.on("click", () => {
            setSelectedZone({
                id: zoneId,
                name: zoneName,
                borough,
                demand,
            });
        });
    }

    return (
        <section className="panel zone-map-panel">
            <div className="panel-heading">
                <div>
                    <h2>Geographic demand map</h2>
                    <p>
                        Explore taxi zones and their GNN-predicted demand.
                    </p>
                </div>

                <span className="map-badge">GNN OUTPUT</span>
            </div>

            {error && (
                <div className="map-error">
                    <strong>Map data could not be loaded</strong>
                    <p>{error}</p>
                    <p>
                        Confirm the file exists at
                        {" "}frontend/public/data/taxi_zones.geojson
                    </p>
                </div>
            )}

            {!geoData && !error && (
                <p className="map-message">
                    Loading taxi-zone boundaries...
                </p>
            )}

            {geoData && (
                <>
                    <div className="map-summary">
                        Showing {geoData.features.length} geographic zones.
                        {" "}{matchedZoneCount} zones match the current predictions.
                    </div>

                    <MapContainer
                        className="zone-demand-map"
                        center={[40.7128, -74.006]}
                        zoom={10}
                        scrollWheelZoom={true}
                    >
                        <TileLayer
                            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        />

                        <FitZoneBounds data={geoData} />

                        <GeoJSON
                            key={`${geoData.features.length}-${predictions.length}`}
                            data={geoData}
                            style={styleFeature}
                            onEachFeature={onEachFeature}
                        />
                    </MapContainer>
                </>
            )}

            <div className="map-legend">
                <span>
                    <i style={{ background: "#10b981" }} />
                    High
                </span>

                <span>
                    <i style={{ background: "#65a30d" }} />
                    Medium
                </span>

                <span>
                    <i style={{ background: "#64748b" }} />
                    Low
                </span>

                <span>
                    <i style={{ background: "#334155" }} />
                    No prediction
                </span>
            </div>

            {selectedZone && (
                <div className="selected-zone">
                    <div>
                        <strong>{selectedZone.name}</strong>
                        <p>
                            Zone ID: {selectedZone.id ?? "Unavailable"}
                            {" "}· Borough: {selectedZone.borough}
                        </p>
                    </div>

                    <div className="selected-zone-value">
                        {selectedZone.demand == null
                            ? "No prediction"
                            : `${selectedZone.demand.toFixed(3)} predicted trips`}
                    </div>

                    <button onClick={() => setSelectedZone(null)}>
                        Close
                    </button>
                </div>
            )}
        </section>
    );
}