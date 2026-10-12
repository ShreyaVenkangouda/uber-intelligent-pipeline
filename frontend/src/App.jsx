import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  MapPin,
  TrendingUp,
  Clock,
  RefreshCw,
  CarFront,
  AlertCircle,
  Search,
  ArrowUpDown,
  Database,
} from 'lucide-react'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts'
import './App.css'
import ZoneDemandMap from "./ZoneDemandMap";

const API_URL = 'http://127.0.0.1:8000'

function formatDateTime(value) {
  if (!value) return 'Unavailable'

  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleString()
}

function formatChartHour(value) {
  if (!value) return ''

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)

  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    hour12: false,
  })
}

function HourlyTooltip({ active, payload }) {
  if (!active || !payload?.length) return null

  const point = payload[0].payload

  return (
    <div className="custom-chart-tooltip">
      <strong>{formatDateTime(point.timestamp)}</strong>
      <p>{Number(point.trips).toLocaleString()} trips</p>
    </div>
  )
}

function App() {
  const [predictionData, setPredictionData] = useState(null)
  const [hourlyData, setHourlyData] = useState([])

  const [loadingPredictions, setLoadingPredictions] = useState(true)
  const [loadingHourly, setLoadingHourly] = useState(true)
  const [predictionError, setPredictionError] = useState('')
  const [hourlyError, setHourlyError] = useState('')
  const [lastUpdated, setLastUpdated] = useState(null)

  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState('predicted_demand')
  const [sortDirection, setSortDirection] = useState('desc')

  const fetchPredictions = useCallback(async () => {
    setLoadingPredictions(true)
    setPredictionError('')

    try {
      const response = await fetch(`${API_URL}/api/predictions`)

      if (!response.ok) {
        const details = await response.json().catch(() => ({}))
        throw new Error(
          details.detail || `Predictions request failed (${response.status})`
        )
      }

      const data = await response.json()
      setPredictionData(data)
      setLastUpdated(new Date())
    } catch (err) {
      setPredictionError(err.message || 'Could not load GNN predictions.')
    } finally {
      setLoadingPredictions(false)
    }
  }, [])

  const fetchHourlyDemand = useCallback(async () => {
    setLoadingHourly(true)
    setHourlyError('')

    try {
      const response = await fetch(`${API_URL}/api/hourly-demand`)

      if (!response.ok) {
        const details = await response.json().catch(() => ({}))
        throw new Error(
          details.detail || `Hourly demand request failed (${response.status})`
        )
      }

      const data = await response.json()

      const formatted = (data.hourly_demand || []).map((item) => ({
        timestamp: item.hour,
        hour: formatChartHour(item.hour),
        trips: Number(item.trips),
      }))

      setHourlyData(formatted)
    } catch (err) {
      setHourlyError(err.message || 'Could not load hourly trip counts.')
    } finally {
      setLoadingHourly(false)
    }
  }, [])

  const refreshAll = useCallback(() => {
    fetchPredictions()
    fetchHourlyDemand()
  }, [fetchPredictions, fetchHourlyDemand])

  useEffect(() => {
    refreshAll()
  }, [refreshAll])

  const predictions = predictionData?.predictions ?? []

  const filteredPredictions = useMemo(() => {
    const term = search.trim().toLowerCase()

    return predictions
      .filter((row) =>
        String(row.pickup_location).includes(term) ||
        String(row.node_index).includes(term) ||
        String(row.predicted_demand).includes(term)
      )
      .sort((a, b) => {
        const aValue = Number(a[sortBy]) || 0
        const bValue = Number(b[sortBy]) || 0

        return sortDirection === 'asc'
          ? aValue - bValue
          : bValue - aValue
      })
  }, [predictions, search, sortBy, sortDirection])

  const topZones = useMemo(
    () => [...predictions]
      .sort(
        (a, b) =>
          Number(b.predicted_demand) - Number(a.predicted_demand)
      )
      .slice(0, 5),
    [predictions]
  )

  const maxDemand = Math.max(
    ...topZones.map((zone) => Number(zone.predicted_demand) || 0),
    1
  )

  const maxHourlyTrips = Math.max(
    ...hourlyData.map((item) => item.trips),
    0
  )

  function changeSort(column) {
    if (sortBy === column) {
      setSortDirection((direction) =>
        direction === 'desc' ? 'asc' : 'desc'
      )
    } else {
      setSortBy(column)
      setSortDirection(
        column === 'predicted_demand' ? 'desc' : 'asc'
      )
    }
  }

  return (
    <main className="dashboard">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <CarFront size={25} />
          </div>
          <div>
            <h1>Uber<span>Pulse</span></h1>
            <p>INTELLIGENT MOBILITY ANALYTICS</p>
          </div>
        </div>

        <div className="topbar-right">
          <span className="status">
            <span className="status-dot" />
            {predictionError || hourlyError
              ? 'Data issue'
              : loadingPredictions || loadingHourly
                ? 'Refreshing data'
                : 'API connected'}
          </span>

          <button
            className="refresh-btn"
            onClick={refreshAll}
            disabled={loadingPredictions || loadingHourly}
          >
            <RefreshCw size={15} />
            {loadingPredictions || loadingHourly ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </header>

      <section className="welcome">
        <div>
          <p className="eyebrow">OPERATIONS OVERVIEW</p>
          <h2>Demand intelligence</h2>
          <p className="subtitle">
            Explore real hourly trip activity and GNN-based demand predictions.
          </p>
        </div>

        <div className="updated">
          {lastUpdated
            ? `Fetched ${lastUpdated.toLocaleTimeString()}`
            : 'Waiting for API'}
        </div>
      </section>

      {(predictionError || hourlyError) && (
        <div className="error-banner">
          <AlertCircle size={19} />
          <div>
            <strong>Some dashboard data could not be loaded</strong>
            {predictionError && <p>Predictions: {predictionError}</p>}
            {hourlyError && <p>Hourly activity: {hourlyError}</p>}
            <button onClick={refreshAll}>Try again</button>
          </div>
        </div>
      )}

      <section className="metrics">
        <article className="metric-card">
          <div className="metric-top">
            <span>Processed trips</span>
            <Activity />
          </div>
          <strong>6,690</strong>
          <p className="metric-note">After deduplication</p>
        </article>

        <article className="metric-card">
          <div className="metric-top">
            <span>Graph locations</span>
            <MapPin />
          </div>
          <strong>220</strong>
          <p className="metric-note">Pickup and drop-off nodes</p>
        </article>

        <article className="metric-card">
          <div className="metric-top">
            <span>Graph connections</span>
            <TrendingUp />
          </div>
          <strong>2,619</strong>
          <p className="metric-note">Directed trip edges</p>
        </article>

        <article className="metric-card">
          <div className="metric-top">
            <span>Prediction horizon</span>
            <Clock />
          </div>
          <strong>+1 hour</strong>
          <p className="metric-note">GNN model output</p>
        </article>
      </section>

      <section className="content-grid">
        <article className="panel chart-panel">
          <div className="panel-heading">
            <div>
              <h3>Hourly trip activity</h3>
              <p>
                {hourlyError
                  ? 'Hourly data unavailable'
                  : `${hourlyData.length} actual hourly records from Spark`}
              </p>
            </div>
            <span className="chart-tag">TRIPS / HOUR</span>
          </div>

          <div className="chart">
            {loadingHourly && hourlyData.length === 0 ? (
              <p className="empty-state">Loading hourly data from Spark...</p>
            ) : hourlyData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={hourlyData}
                  margin={{ top: 10, right: 8, left: -18, bottom: 0 }}
                >
                  <defs>
                    <linearGradient
                      id="demandFill"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop offset="0%" stopColor="#36e0a1" stopOpacity={0.30} />
                      <stop offset="100%" stopColor="#36e0a1" stopOpacity={0.01} />
                    </linearGradient>
                  </defs>

                  <CartesianGrid strokeDasharray="3 3" stroke="#263248" />

                  <XAxis
                    dataKey="hour"
                    interval="preserveStartEnd"
                    minTickGap={24}
                    tick={{ fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />

                  <YAxis
                    domain={[0, 'auto']}
                    tick={{ fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    allowDecimals={false}
                  />

                  <Tooltip content={<HourlyTooltip />} />

                  <Area
                    type="monotone"
                    dataKey="trips"
                    stroke="#36e0a1"
                    strokeWidth={2.5}
                    fill="url(#demandFill)"
                    activeDot={{ r: 5, strokeWidth: 0 }}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <p className="empty-state">No hourly data available.</p>
            )}
          </div>

          <p className="panel-footnote">
            {hourlyData.length
              ? `Peak observed hour: ${hourlyData.reduce(
                (best, item) => item.trips > best.trips ? item : best,
                hourlyData[0]
              ).trips.toLocaleString()} trips`
              : 'Waiting for hourly records'}
            {maxHourlyTrips > 0 ? ` · Maximum: ${maxHourlyTrips.toLocaleString()}` : ''}
          </p>
        </article>

        <article className="panel zones-panel">
          <div className="panel-heading">
            <div>
              <h3>Predicted demand</h3>
              <p>
                {predictionData
                  ? `${predictionData.count} zones scored by the model`
                  : 'Waiting for model output'}
              </p>
            </div>
            <span className="live-tag">GNN OUTPUT</span>
          </div>

          {loadingPredictions && !predictionData ? (
            <p className="empty-state">Loading predictions from MinIO...</p>
          ) : topZones.length > 0 ? (
            <div className="zone-list">
              {topZones.map((zone, index) => {
                const demand = Number(zone.predicted_demand) || 0

                return (
                  <div className="zone-row" key={zone.pickup_location}>
                    <span className="zone-rank">
                      {String(index + 1).padStart(2, '0')}
                    </span>

                    <div className="zone-info">
                      <div className="zone-label">
                        <strong>Zone {zone.pickup_location}</strong>
                        <span>{demand.toFixed(2)} predicted trips</span>
                      </div>

                      <div className="bar-track">
                        <div
                          className="bar-fill"
                          style={{ width: `${demand / maxDemand * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <p className="empty-state">No predictions available.</p>
          )}

          <p className="panel-footnote">
            Prediction hour: {formatDateTime(predictionData?.prediction_hour)}
          </p>
        </article>
      </section>

      <section className="panel predictions-panel">
        <div className="panel-heading predictions-heading">
          <div>
            <h3>Zone predictions explorer</h3>
            <p>Search and sort the predictions returned by the API.</p>
          </div>
          <span className="live-tag">
            <Database size={12} /> MINIO DATA
          </span>
        </div>

        <div className="table-toolbar">
          <label className="search-box">
            <Search size={17} />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search zone, node, or demand..."
              aria-label="Search predictions"
            />
          </label>

          <span className="result-count">
            {filteredPredictions.length} of {predictions.length} records
          </span>
        </div>

        <div className="table-scroll">
          <table className="predictions-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>
                  <button
                    className="sort-button"
                    onClick={() => changeSort('pickup_location')}
                  >
                    Pickup zone <ArrowUpDown size={13} />
                  </button>
                </th>
                <th>
                  <button
                    className="sort-button"
                    onClick={() => changeSort('node_index')}
                  >
                    Graph node <ArrowUpDown size={13} />
                  </button>
                </th>
                <th>
                  <button
                    className="sort-button"
                    onClick={() => changeSort('predicted_demand')}
                  >
                    Predicted demand <ArrowUpDown size={13} />
                  </button>
                </th>
                <th>Prediction hour</th>
              </tr>
            </thead>

            <tbody>
              {loadingPredictions && !predictionData ? (
                <tr>
                  <td colSpan={5} className="table-message">
                    Loading predictions from the API...
                  </td>
                </tr>
              ) : filteredPredictions.length > 0 ? (
                filteredPredictions.map((row, index) => (
                  <tr
                    key={`${row.pickup_location}-${row.node_index}`}
                  >
                    <td>
                      <span className="table-rank">
                        {String(index + 1).padStart(2, '0')}
                      </span>
                    </td>
                    <td>
                      <strong className="table-zone">
                        Zone {row.pickup_location}
                      </strong>
                    </td>
                    <td>{row.node_index}</td>
                    <td>
                      <span className="demand-value">
                        {Number(row.predicted_demand).toFixed(3)}
                      </span>
                    </td>
                    <td className="table-date">
                      {formatDateTime(row.prediction_hour)}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="table-message">
                    No matching predictions found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <p className="panel-footnote">
          Historical model output · Prediction hour:{' '}
          {formatDateTime(predictionData?.prediction_hour)}
        </p>
      </section>

      <section className="panel zone-map-wrapper">
        <ZoneDemandMap predictions={predictions} />
      </section>

      <footer>
        <span>
          <span className="status-dot" />
          UberPulse Analytics
        </span>
        <span>Kafka · MinIO · Spark · PyTorch Geometric</span>
      </footer>
    </main>
  )
}

export default App
