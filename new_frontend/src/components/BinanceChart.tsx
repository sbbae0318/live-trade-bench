import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import './BinanceChart.css';

// Crosshair 플러그인 (버티컬 라인) - 전문 용어: Crosshair 또는 Cursor Line
const crosshairPlugin = {
  id: 'crosshair',
  afterDraw: (chart: any) => {
    if (chart.tooltip && chart.tooltip._active && chart.tooltip._active.length > 0) {
      const ctx = chart.ctx;
      const activePoint = chart.tooltip._active[0];
      const x = activePoint.element.x;
      const yAxis = chart.scales.y;
      
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(x, yAxis.top);
      ctx.lineTo(x, yAxis.bottom);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(25, 118, 210, 0.5)';
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.restore();
    }
  }
};

// Chart.js 등록
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  zoomPlugin,
  crosshairPlugin
);

interface ChartDataPoint {
  timestamp: string;
  close_time: number;
  price: number;
  quantity: number;
  allocation: number;
  position_value: number;
}

interface ChartData {
  [symbol: string]: ChartDataPoint[];
}

interface TotalValueDataPoint {
  timestamp: string;
  close_time: number;
  total_value: number;
}

interface BinanceChartProps {
  agentName?: string;
  symbols?: string[];
  updateInterval?: number; // 초 단위
}

const BinanceChart: React.FC<BinanceChartProps> = ({
  agentName,
  symbols,
  updateInterval = 30,
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'details'>('overview'); // 탭 상태
  const overviewChartRef = React.useRef<any>(null);
  
  // Overview 탭용 상태 (모든 에이전트의 total value)
  const [agents, setAgents] = useState<string[]>([]);
  const [allAgentsTotalValue, setAllAgentsTotalValue] = useState<{[agent: string]: TotalValueDataPoint[]}>({});
  const [overviewLoading, setOverviewLoading] = useState<boolean>(true);
  const [overviewUpdating, setOverviewUpdating] = useState<boolean>(false);
  
  // Details 탭용 상태 (기존 심볼별 차트)
  const [selectedAgent, setSelectedAgent] = useState<string>(agentName || '');
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(symbols || []);
  const [chartData, setChartData] = useState<ChartData>({});
  const [totalValueData, setTotalValueData] = useState<TotalValueDataPoint[]>([]);
  const [detailsLoading, setDetailsLoading] = useState<boolean>(true);
  const [detailsUpdating, setDetailsUpdating] = useState<boolean>(false);
  
  // 공통 상태
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [timeUntilNextUpdate, setTimeUntilNextUpdate] = useState<number>(updateInterval);
  
  // 시간 범위 프리셋 상태
  const [timeRangePreset, setTimeRangePreset] = useState<'1day' | '1week' | '1month' | '1year'>('1month');
  
  // 스크롤바 상태 (0-100, 현재 보이는 범위의 시작 위치) - 기본값은 맨 오른쪽(최신)
  const [scrollPosition, setScrollPosition] = useState<number>(100);
  
  // 전체 데이터 범위
  const [dataTimeRange, setDataTimeRange] = useState<{min: number, max: number} | null>(null);
  
  // 보이는 시간 범위 계산 (useMemo로 메모이제이션)
  const visibleTimeRange = useMemo(() => {
    if (!dataTimeRange) {
      return { min: 0, max: 0 };
    }
    
    const presetRanges: { [key: string]: number } = {
      '1day': 24 * 60 * 60 * 1000,
      '1week': 7 * 24 * 60 * 60 * 1000,
      '1month': 30 * 24 * 60 * 60 * 1000,
      '1year': 365 * 24 * 60 * 60 * 1000,
    };
    const visibleRange = presetRanges[timeRangePreset];
    const totalRange = dataTimeRange.max - dataTimeRange.min;
    const scrollableRange = Math.max(0, totalRange - visibleRange);
    
    const startTime = dataTimeRange.min + (scrollableRange * scrollPosition / 100);
    const newMin = Math.max(dataTimeRange.min, Math.min(dataTimeRange.max - visibleRange, startTime));
    
    return { min: newMin, max: newMin + visibleRange };
  }, [dataTimeRange, timeRangePreset, scrollPosition]);
  
  // 포트폴리오 비중 상태
  const [portfolioWeights, setPortfolioWeights] = useState<{[agent: string]: {
    symbols: {[symbol: string]: {
      allocation: number;
      actual_weight: number;
      position_value: number;
      current_price: number;
      quantity: number;
    }};
    total_value: number;
  }}>({});
  const [portfolioLoading, setPortfolioLoading] = useState<boolean>(true);

  // Snapshot Modal 상태
  const [snapshotModalOpen, setSnapshotModalOpen] = useState<boolean>(false);
  const [snapshotData, setSnapshotData] = useState<{
    close_time: number;
    timestamp: string | null;
    snapshot: {
      [agent: string]: {
        timestamp: string | null;
        close_time: number;
        last_trade_time: string | null;
        symbols: {
          [symbol: string]: {
            allocation: number;
            current_price: number;
            quantity: number;
            position_value: number;
            average_price: number;
          };
        };
      };
    };
  } | null>(null);
  const [snapshotLoading, setSnapshotLoading] = useState<boolean>(false);

  // API 기본 URL
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

  // 에이전트 목록 가져오기
  const fetchAgents = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/binance/chart/agents`);
      if (!response.ok) {
        throw new Error('Failed to fetch agents');
      }
      const data = await response.json();
      setAgents(data.agents || []);
      
      // 첫 번째 에이전트 선택 (Details 탭용)
      if (!selectedAgent && data.agents && data.agents.length > 0) {
        setSelectedAgent(data.agents[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch agents');
    }
  }, [selectedAgent, API_BASE_URL]);

  // 포트폴리오 비중 가져오기
  const fetchPortfolioWeights = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/binance/chart/portfolio-weights`);
      if (!response.ok) {
        throw new Error('Failed to fetch portfolio weights');
      }
      const data = await response.json();
      setPortfolioWeights(data.portfolio_weights || {});
      setPortfolioLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch portfolio weights');
      setPortfolioLoading(false);
    }
  }, [API_BASE_URL]);

  // Overview 탭: 모든 에이전트의 total value 가져오기
  const fetchAllAgentsTotalValue = useCallback(async (isInitial: boolean = false) => {
    if (agents.length === 0) {
      return;
    }

    if (!isInitial) {
      setOverviewUpdating(true);
    } else {
      setOverviewLoading(true);
    }
    setError(null);

    try {
      // 모든 에이전트의 total value를 병렬로 가져오기
      const responses = await Promise.all(
        agents.map(agent => 
          fetch(`${API_BASE_URL}/api/binance/chart/${agent}/total-value`)
            .then(res => res.ok ? res.json() : null)
            .catch(() => null)
        )
      );

      const agentsData: {[agent: string]: TotalValueDataPoint[]} = {};
      responses.forEach((data, index) => {
        if (data && data.data) {
          agentsData[agents[index]] = data.data;
        }
      });

      setAllAgentsTotalValue(agentsData);
      const now = new Date();
      setLastUpdate(now);
      setTimeUntilNextUpdate(updateInterval);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch all agents total value');
    } finally {
      if (isInitial) {
        setOverviewLoading(false);
      } else {
        setOverviewUpdating(false);
      }
    }
  }, [agents, API_BASE_URL]);

  // 심볼 목록 가져오기
  const fetchSymbols = useCallback(async (agent: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/binance/chart/${agent}/symbols`);
      if (!response.ok) {
        throw new Error('Failed to fetch symbols');
      }
      const data = await response.json();
      setAvailableSymbols(data.symbols || []);
      
      // 선택된 심볼이 없으면 모든 심볼 선택
      if (selectedSymbols.length === 0 && data.symbols && data.symbols.length > 0) {
        setSelectedSymbols(data.symbols);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch symbols');
    }
  }, [selectedSymbols, API_BASE_URL]);

  // 차트 데이터 가져오기
  const fetchChartData = useCallback(async (agent: string, symbolsList: string[], isInitial: boolean = false) => {
    if (!agent || symbolsList.length === 0) {
      return;
    }

      // 초기 로딩이 아닌 경우에만 updating 상태 설정 (화면 깜빡임 방지)
    if (!isInitial) {
      setDetailsUpdating(true);
    } else {
      setDetailsLoading(true);
    }
    setError(null);

    try {
      const symbolsParam = symbolsList.join(',');
      
      // 심볼별 차트 데이터와 total 가치 데이터를 병렬로 가져오기
      const [chartResponse, totalValueResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/binance/chart/${agent}/data?symbols=${symbolsParam}&include_realtime=true`),
        fetch(`${API_BASE_URL}/api/binance/chart/${agent}/total-value`)
      ]);
      
      if (!chartResponse.ok) {
        throw new Error('Failed to fetch chart data');
      }
      
      if (!totalValueResponse.ok) {
        throw new Error('Failed to fetch total value data');
      }
      
      const chartDataResult = await chartResponse.json();
      const totalValueResult = await totalValueResponse.json();
      
      // 데이터 업데이트를 배치로 처리하여 렌더링 최적화
      setChartData(prevData => {
        // 기존 데이터와 새 데이터를 병합 (스크롤 위치 유지)
        return { ...prevData, ...chartDataResult.data };
      });
      
      setTotalValueData(totalValueResult.data || []);
      const now = new Date();
      setLastUpdate(now);
      setTimeUntilNextUpdate(updateInterval);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch chart data');
    } finally {
      if (isInitial) {
        setDetailsLoading(false);
      } else {
        setDetailsUpdating(false);
      }
    }
  }, [API_BASE_URL]);

  // 초기 로드: 에이전트 목록 가져오기
  useEffect(() => {
    fetchAgents();
    fetchPortfolioWeights();
  }, [fetchAgents, fetchPortfolioWeights]);

  // 포트폴리오 비중 주기적 업데이트
  useEffect(() => {
    const interval = setInterval(() => {
      fetchPortfolioWeights();
    }, updateInterval * 1000);

    return () => clearInterval(interval);
  }, [fetchPortfolioWeights, updateInterval]);

  // Overview 탭: 에이전트 목록이 로드되면 total value 가져오기
  useEffect(() => {
    if (agents.length > 0) {
      fetchAllAgentsTotalValue(true);
    }
  }, [agents.length]); // agents.length만 의존성으로 사용

  // Overview 탭: 주기적 업데이트
  useEffect(() => {
    if (activeTab !== 'overview' || agents.length === 0) {
      return;
    }

    const interval = setInterval(() => {
      fetchAllAgentsTotalValue(false);
    }, updateInterval * 1000);

    return () => clearInterval(interval);
  }, [activeTab, agents.length, updateInterval, fetchAllAgentsTotalValue]);

  // 에이전트 변경 시 심볼 가져오기
  useEffect(() => {
    if (selectedAgent) {
      fetchSymbols(selectedAgent);
    }
  }, [selectedAgent, fetchSymbols]);

  // Details 탭: 차트 데이터 업데이트 (초기 로드)
  useEffect(() => {
    if (activeTab === 'details' && selectedAgent && selectedSymbols.length > 0) {
      fetchChartData(selectedAgent, selectedSymbols, true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, selectedAgent, selectedSymbols.join(',')]); // activeTab 추가

  // Details 탭: 주기적 업데이트
  useEffect(() => {
    if (activeTab !== 'details' || !selectedAgent || selectedSymbols.length === 0 || detailsLoading) {
      return;
    }

    const interval = setInterval(() => {
      fetchChartData(selectedAgent, selectedSymbols, false);
    }, updateInterval * 1000);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, selectedAgent, selectedSymbols.join(','), updateInterval, detailsLoading]);

  // 다음 업데이트까지 남은 시간 계산 (1초마다 업데이트)
  useEffect(() => {
    if (!lastUpdate) {
      return;
    }

    const timer = setInterval(() => {
      const now = new Date();
      const elapsed = Math.floor((now.getTime() - lastUpdate.getTime()) / 1000);
      const remaining = Math.max(0, updateInterval - elapsed);
      setTimeUntilNextUpdate(remaining);
    }, 1000);

    return () => clearInterval(timer);
  }, [lastUpdate, updateInterval]);

  // 차트 옵션 설정
  const getChartOptions = (symbol: string) => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      elements: {
        point: {
          radius: 0,
          hoverRadius: 0,
          borderWidth: 0,
          backgroundColor: 'transparent',
        },
      },
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: `${symbol} - Price History`,
        },
        tooltip: {
          mode: 'index' as const,
          intersect: false,
          callbacks: {
            label: function(context: any) {
              const point = chartData[symbol]?.[context.dataIndex];
              if (!point) return '';
              
              if (context.datasetIndex === 0) {
                return `Price: $${point.price.toFixed(4)}`;
              } else {
                return `Position Value: $${point.position_value.toFixed(2)}`;
              }
            },
            afterBody: function(context: any) {
              const point = chartData[symbol]?.[context[0]?.dataIndex];
              if (!point) return '';
              
              return [
                `Quantity: ${point.quantity.toFixed(4)}`,
                `Allocation: ${(point.allocation * 100).toFixed(2)}%`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          type: 'time' as const,
          time: {
            unit: 'minute' as const,
            displayFormats: {
              minute: 'HH:mm',
            },
          },
          title: {
            display: true,
            text: 'Time',
          },
        },
        y: {
          type: 'linear' as const,
          position: 'left' as const,
          title: {
            display: true,
            text: 'Price (USDT)',
          },
        },
        y1: {
          type: 'linear' as const,
          position: 'right' as const,
          title: {
            display: true,
            text: 'Position Value (USDT)',
          },
          grid: {
            drawOnChartArea: false,
          },
        },
      },
    };
  };

  // 차트 데이터 변환
  const getChartDataForSymbol = (symbol: string) => {
    const dataPoints = chartData[symbol] || [];
    
    return {
      datasets: [
        {
          label: 'Price',
          data: dataPoints.map((point) => ({
            x: new Date(point.timestamp),
            y: point.price,
          })),
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.1,
          borderWidth: 1,
          pointRadius: 0,
          pointHoverRadius: 0,
          pointBorderWidth: 0,
          pointBackgroundColor: 'transparent',
          yAxisID: 'y',
        },
        {
          label: 'Position Value',
          data: dataPoints.map((point) => ({
            x: new Date(point.timestamp),
            y: point.position_value,
          })),
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.1,
          borderWidth: 1,
          pointRadius: 0,
          pointHoverRadius: 0,
          pointBorderWidth: 0,
          pointBackgroundColor: 'transparent',
          yAxisID: 'y1',
        },
      ],
    };
  };

  // Total Value 차트 데이터 변환
  const getTotalValueChartData = () => {
    return {
      datasets: [
        {
          label: 'Total Value',
          data: totalValueData.map((point) => ({
            x: new Date(point.timestamp),
            y: point.total_value,
          })),
          borderColor: 'rgb(54, 162, 235)',
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          tension: 0.1,
          borderWidth: 1,
          pointRadius: 0,
          pointHoverRadius: 0,
          pointBorderWidth: 0,
          pointBackgroundColor: 'transparent',
          fill: true,
        },
      ],
    };
  };

  // 전체 데이터 범위 계산 (useEffect로 분리하여 무한 루프 방지)
  useEffect(() => {
    let allTimestamps: number[] = [];
    
    agents.forEach(agent => {
      const data = allAgentsTotalValue[agent] || [];
      data.forEach(point => {
        const timestamp = new Date(point.timestamp).getTime();
        if (!allTimestamps.includes(timestamp)) {
          allTimestamps.push(timestamp);
        }
      });
    });
    
    if (allTimestamps.length > 0) {
      const minTime = Math.min(...allTimestamps);
      const maxTime = Math.max(...allTimestamps);
      const newRange = { min: minTime, max: maxTime };
      
      // 기존 범위와 다를 때만 업데이트
      setDataTimeRange(prevRange => {
        if (!prevRange || prevRange.min !== minTime || prevRange.max !== maxTime) {
          return newRange;
        }
        return prevRange;
      });
    }
  }, [agents, allAgentsTotalValue]);

  // Overview 탭: 모든 에이전트의 total value 차트 데이터
  const getAllAgentsTotalValueChartData = () => {
    const datasets = agents.map((agent, index) => {
      const data = allAgentsTotalValue[agent] || [];
      const colors = [
        'rgb(54, 162, 235)',
        'rgb(255, 99, 132)',
        'rgb(75, 192, 192)',
        'rgb(255, 159, 64)',
        'rgb(153, 102, 255)',
        'rgb(201, 203, 207)',
      ];
      const color = colors[index % colors.length];
      
      return {
        label: agent,
        data: data.map((point) => ({
          x: new Date(point.timestamp),
          y: point.total_value,
          close_time: point.close_time, // 클릭 이벤트에서 사용하기 위해 추가
        })),
        borderColor: color,
        backgroundColor: color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
        tension: 0.1,
        borderWidth: 1,
        pointRadius: 0,
        pointHoverRadius: 0,
        pointBorderWidth: 0,
        pointBackgroundColor: 'transparent',
        fill: false,
      };
    });

    return { datasets };
  };

  // Overview 탭: 모든 에이전트의 total value 차트 옵션
  const getAllAgentsTotalValueChartOptions = () => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      // 성능 최적화: 애니메이션 비활성화 (Chart.js v4 형식)
      animation: {
        duration: 0,
      },
      // 성능 최적화: 상호작용 최적화
      interaction: {
        mode: 'index' as const,
        intersect: false,
        // 성능 최적화: hover 시에만 tooltip 표시
        includeInvisible: false,
      },
      elements: {
        point: {
          radius: 0,
          hoverRadius: 0,
          borderWidth: 0,
          backgroundColor: 'transparent',
        },
      },
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: 'All Agents - Total Portfolio Value Comparison',
        },
        tooltip: {
          // tooltip의 mode와 intersect는 최상위 interaction 설정을 따름
          // 성능 최적화: tooltip 애니메이션 비활성화
          animation: {
            duration: 0,
          },
          callbacks: {
            label: function(context: any) {
              return `${context.dataset.label}: $${context.parsed.y.toFixed(2)}`;
            },
          },
        },
        zoom: {
          zoom: {
            wheel: {
              enabled: false, // 휠 줌 비활성화
            },
            pinch: {
              enabled: false, // 핀치 줌 비활성화
            },
            mode: 'x' as const,
            animation: {
              duration: 0,
            },
          },
          pan: {
            enabled: false, // 팬 비활성화
          },
        },
      },
      scales: {
        x: {
          type: 'time' as const,
          time: {
            unit: 'day' as const,
            displayFormats: {
              day: 'MMM dd',
            },
          },
          title: {
            display: true,
            text: 'Date',
          },
          // 시간 범위 프리셋과 스크롤바 위치에 따른 min/max 설정
          min: dataTimeRange ? visibleTimeRange.min : (() => {
            const now = Date.now();
            const ranges: { [key: string]: number } = {
              '1day': now - 24 * 60 * 60 * 1000,
              '1week': now - 7 * 24 * 60 * 60 * 1000,
              '1month': now - 30 * 24 * 60 * 60 * 1000,
              '1year': now - 365 * 24 * 60 * 60 * 1000,
            };
            return ranges[timeRangePreset];
          })(),
          max: dataTimeRange ? visibleTimeRange.max : Date.now(),
        },
        y: {
          type: 'linear' as const,
          position: 'left' as const,
          title: {
            display: true,
            text: 'Total Value (USDT)',
          },
        },
      },
      onClick: (event: any, elements: any[]) => {
        const chart = event.chart;
        if (!chart || !chart.canvas) return;
        
        const canvas = chart.canvas;
        const rect = canvas.getBoundingClientRect();
        const nativeEvent = event.native as MouseEvent | null;
        
        if (!nativeEvent) return;
        
        const x = nativeEvent.clientX - rect.left;
        
        // 클릭한 x 좌표에 해당하는 시간값 찾기
        const xValue = chart.scales.x.getValueForPixel(x);
        
        if (xValue) {
          // xValue를 Date 객체로 변환 (이미 Date이거나 숫자일 수 있음)
          const clickedTime = xValue instanceof Date ? xValue.getTime() : new Date(xValue).getTime();
          
          // 모든 데이터셋에서 해당 시간에 가장 가까운 데이터 포인트 찾기
          let closestCloseTime: number | null = null;
          let minTimeDiff = Infinity;
          
          chart.data.datasets.forEach((dataset: any) => {
            dataset.data.forEach((point: any) => {
              if (point && point.x && point.close_time) {
                // point.x도 Date 객체 또는 숫자일 수 있음
                const pointTime = point.x instanceof Date ? point.x.getTime() : new Date(point.x).getTime();
                const timeDiff = Math.abs(pointTime - clickedTime);
                if (timeDiff < minTimeDiff) {
                  minTimeDiff = timeDiff;
                  closestCloseTime = point.close_time;
                }
              }
            });
          });
          
          // 5분(300000ms) 이내의 차이면 해당 시점으로 간주
          if (closestCloseTime && minTimeDiff < 300000) {
            fetchSnapshotData(closestCloseTime);
          }
        }
      },
      onHover: (event: any, elements: any[]) => {
        // 마우스 커서를 포인터로 변경
        if (event.native) {
          event.native.target.style.cursor = 'pointer';
        }
      },
    };
  };

  // Details 탭: Total Value 차트 옵션
  // Snapshot 데이터 가져오기
  const fetchSnapshotData = useCallback(async (closeTime: number) => {
    setSnapshotLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/binance/chart/snapshot/${closeTime}`);
      if (!response.ok) {
        throw new Error('Failed to fetch snapshot data');
      }
      const data = await response.json();
      setSnapshotData(data);
      setSnapshotModalOpen(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch snapshot data');
    } finally {
      setSnapshotLoading(false);
    }
  }, [API_BASE_URL]);

  const getTotalValueChartOptions = () => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      elements: {
        point: {
          radius: 0,
          hoverRadius: 0,
          borderWidth: 0,
          backgroundColor: 'transparent',
        },
      },
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: 'Total Portfolio Value',
        },
        tooltip: {
          mode: 'index' as const,
          intersect: false,
          callbacks: {
            label: function(context: any) {
              return `Total Value: $${context.parsed.y.toFixed(2)}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: 'time' as const,
          time: {
            unit: 'day' as const,
            displayFormats: {
              day: 'MMM dd',
            },
          },
          title: {
            display: true,
            text: 'Date',
          },
        },
        y: {
          type: 'linear' as const,
          position: 'left' as const,
          title: {
            display: true,
            text: 'Total Value (USDT)',
          },
        },
      },
      onClick: (event: any, elements: any[]) => {
        if (elements.length > 0) {
          const element = elements[0];
          const chart = event.chart;
          const datasetIndex = element.datasetIndex;
          const index = element.index;
          
          // 해당 데이터 포인트의 close_time 가져오기
          const dataset = chart.data.datasets[datasetIndex];
          const dataPoint = dataset.data[index];
          
          if (dataPoint && dataPoint.close_time) {
            fetchSnapshotData(dataPoint.close_time);
          }
        }
      },
      onHover: (event: any, elements: any[]) => {
        // 마우스 커서를 포인터로 변경
        if (event.native) {
          event.native.target.style.cursor = elements.length > 0 ? 'pointer' : 'default';
        }
      },
    };
  };

  return (
    <div className="binance-chart-container">
      {/* 탭 메뉴 */}
      <div className="chart-tabs">
        <button
          className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={`tab-button ${activeTab === 'details' ? 'active' : ''}`}
          onClick={() => setActiveTab('details')}
        >
          Details
        </button>
      </div>

      {/* Overview 탭 */}
      {activeTab === 'overview' && (
        <div className="tab-content">
          {overviewLoading && (
            <div className="loading-message">Loading overview data...</div>
          )}

          {overviewUpdating && !overviewLoading && (
            <div className="updating-indicator">Updating...</div>
          )}

          {lastUpdate && activeTab === 'overview' && (
            <div className="update-status-overview">
              <div className="last-update-overview">
                Last update: {lastUpdate.toLocaleTimeString()}
              </div>
              <div className="next-update-countdown-overview">
                <div className="countdown-label">Next update in:</div>
                <div className="countdown-timer">
                  <span className="countdown-value">{timeUntilNextUpdate}</span>
                  <span className="countdown-unit">s</span>
                </div>
                <div className="countdown-progress-bar">
                  <div 
                    className="countdown-progress-fill"
                    style={{ width: `${((updateInterval - timeUntilNextUpdate) / updateInterval) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {!overviewLoading && agents.length > 0 && (
            <div className="overview-layout">
              <div className="overview-chart-container">
                <div className="chart-zoom-controls">
                  <div className="zoom-controls-right">
                    <select
                      className="time-range-select"
                      value={timeRangePreset}
                      onChange={(e) => {
                        setTimeRangePreset(e.target.value as '1day' | '1week' | '1month' | '1year');
                        setScrollPosition(100); // 프리셋 변경 시 스크롤바 위치를 맨 오른쪽(최신)으로 설정
                        // 차트 업데이트를 위해 강제 리렌더링
                        if (overviewChartRef.current) {
                          const chart = overviewChartRef.current;
                          if (chart && chart.update) {
                            chart.update('none');
                          }
                        }
                      }}
                    >
                      <option value="1day">하루</option>
                      <option value="1week">일주일</option>
                      <option value="1month">한달</option>
                      <option value="1year">1년</option>
                    </select>
                  </div>
                  <span className="zoom-hint">아래 스크롤바로 시간 범위 이동</span>
                </div>
              <div className="chart-wrapper overview-chart">
                <Line
                    ref={overviewChartRef}
                  data={getAllAgentsTotalValueChartData()}
                  options={getAllAgentsTotalValueChartOptions()}
                  // 성능 최적화: 차트 업데이트 모드 설정
                  updateMode="none"
                  redraw={false}
                />
                </div>
                {/* 스크롤바 */}
                {dataTimeRange && (() => {
                  const presetRanges: { [key: string]: number } = {
                    '1day': 24 * 60 * 60 * 1000,
                    '1week': 7 * 24 * 60 * 60 * 1000,
                    '1month': 30 * 24 * 60 * 60 * 1000,
                    '1year': 365 * 24 * 60 * 60 * 1000,
                  };
                  const visibleRange = presetRanges[timeRangePreset];
                  const totalRange = dataTimeRange.max - dataTimeRange.min;
                  const scrollableRange = Math.max(0, totalRange - visibleRange);
                  const showScrollbar = scrollableRange > 0;
                  
                  return (
                    <div className="chart-scrollbar-container">
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={scrollPosition}
                        onChange={(e) => {
                          setScrollPosition(Number(e.target.value));
                          if (overviewChartRef.current && overviewChartRef.current.update) {
                            overviewChartRef.current.update('none');
                          }
                        }}
                        className="chart-scrollbar"
                        disabled={!showScrollbar}
                        style={{ opacity: showScrollbar ? 1 : 0.5 }}
                      />
                    </div>
                  );
                })()}
              </div>
              <div className="portfolio-weights-container">
                <h3 className="portfolio-weights-title">Portfolio Weights</h3>
                {portfolioLoading ? (
                  <div className="loading-message">Loading portfolio weights...</div>
                ) : (
                  <div className="portfolio-weights-table-container">
                    {Object.entries(portfolioWeights).map(([agent, data]) => (
                      <div key={agent} className="portfolio-agent-section">
                        <h4 className="portfolio-agent-name">{agent}</h4>
                        <div className="portfolio-total-value">
                          Total: ${data.total_value.toFixed(2)}
                        </div>
                        <table className="portfolio-table">
                          <thead>
                            <tr>
                              <th>Symbol</th>
                              <th>Target</th>
                              <th>Actual</th>
                              <th>Value</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(data.symbols)
                              .sort(([, a], [, b]) => b.actual_weight - a.actual_weight)
                              .map(([symbol, info]) => (
                                <tr key={symbol}>
                                  <td className="symbol-cell">{symbol}</td>
                                  <td className="weight-cell">{info.allocation.toFixed(1)}%</td>
                                  <td className={`weight-cell ${Math.abs(info.allocation - info.actual_weight) > 1 ? 'weight-diff' : ''}`}>
                                    {info.actual_weight.toFixed(1)}%
                                  </td>
                                  <td className="value-cell">${info.position_value.toFixed(2)}</td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {!overviewLoading && agents.length === 0 && (
            <div className="no-data-message">
              No agents available.
            </div>
          )}
        </div>
      )}

      {/* Details 탭 */}
      {activeTab === 'details' && (
        <div className="tab-content">
          <div className="binance-chart-controls">
        <div className="control-group">
          <label htmlFor="agent-select">Agent:</label>
          <select
            id="agent-select"
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
          >
            <option value="">Select Agent</option>
            {agents.map((agent) => (
              <option key={agent} value={agent}>
                {agent}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Symbols:</label>
          <div className="symbol-checkboxes">
            {availableSymbols.map((symbol) => (
              <label key={symbol} className="checkbox-label">
                <input
                  type="checkbox"
                  checked={selectedSymbols.includes(symbol)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedSymbols([...selectedSymbols, symbol]);
                    } else {
                      setSelectedSymbols(selectedSymbols.filter((s) => s !== symbol));
                    }
                  }}
                />
                {symbol}
              </label>
            ))}
          </div>
        </div>

            {lastUpdate && (
              <div className="update-status">
              <div className="last-update">
                Last update: {lastUpdate.toLocaleTimeString()}
                </div>
                <div className="next-update-countdown">
                  <div className="countdown-label">Next update in:</div>
                  <div className="countdown-timer">
                    <span className="countdown-value">{timeUntilNextUpdate}</span>
                    <span className="countdown-unit">s</span>
                  </div>
                  <div className="countdown-progress-bar">
                    <div 
                      className="countdown-progress-fill"
                      style={{ width: `${((updateInterval - timeUntilNextUpdate) / updateInterval) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {error && <div className="error-message">Error: {error}</div>}

          {/* 초기 로딩만 전체 화면 표시 */}
          {detailsLoading && (
            <div className="loading-message">Loading chart data...</div>
          )}

          {/* 업데이트 중일 때는 작은 인디케이터만 표시 */}
          {detailsUpdating && !detailsLoading && (
            <div className="updating-indicator">Updating...</div>
          )}

          {/* 초기 로딩이 완료되면 차트 표시 (스크롤 위치 유지) */}
          {!detailsLoading && (
            <div className="chart-grid">
              {/* Total Value 차트 */}
              {totalValueData.length > 0 && (
                <div className="chart-wrapper total-value-chart">
                  <Line
                    data={getTotalValueChartData()}
                    options={getTotalValueChartOptions()}
                    updateMode="none"
                    redraw={false}
                  />
                </div>
              )}

              {/* 심볼별 차트 */}
              {selectedSymbols.length > 0 && selectedSymbols.map((symbol) => {
            const symbolData = chartData[symbol];
            if (!symbolData || symbolData.length === 0) {
              return (
                <div key={symbol} className="chart-placeholder">
                  No data available for {symbol}
                </div>
              );
            }

            return (
              <div key={symbol} className="chart-wrapper">
                <Line
                  data={getChartDataForSymbol(symbol)}
                  options={getChartOptions(symbol)}
                  updateMode="none" // 자동 업데이트 비활성화
                  redraw={false} // 재생성하지 않고 업데이트만
                />
              </div>
            );
          })}
        </div>
      )}

          {!detailsLoading && selectedSymbols.length === 0 && totalValueData.length === 0 && (
            <div className="no-data-message">
              Please select at least one symbol to display charts.
            </div>
          )}
        </div>
      )}

      {/* Snapshot Modal */}
      {snapshotModalOpen && (
        <div className="modal-overlay" onClick={() => setSnapshotModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Snapshot at Selected Time</h2>
              <button className="modal-close" onClick={() => setSnapshotModalOpen(false)}>
                ×
              </button>
            </div>
            <div className="modal-body">
              {snapshotLoading ? (
                <div className="loading-message">Loading snapshot data...</div>
              ) : snapshotData ? (
                <>
                  <div className="snapshot-info">
                    <p><strong>Timestamp:</strong> {snapshotData.timestamp ? new Date(snapshotData.timestamp).toLocaleString() : new Date(snapshotData.close_time).toLocaleString()}</p>
                    <p><strong>Close Time:</strong> {snapshotData.close_time}</p>
                  </div>
                  <div className="snapshot-agents">
                    {Object.entries(snapshotData.snapshot).map(([agentName, agentData]) => {
                      const symbols = Object.entries(agentData.symbols);
                      const totalAllocation = symbols.reduce((sum, [, data]) => sum + data.allocation, 0);
                      
                      return (
                        <div key={agentName} className="snapshot-agent-section">
                          <h3>{agentName}</h3>
                          <div className="snapshot-agent-info">
                            <p><strong>Timestamp:</strong> {agentData.timestamp ? new Date(agentData.timestamp).toLocaleString() : 'N/A'}</p>
                            <p><strong>Last Trade Time:</strong> {agentData.last_trade_time ? new Date(agentData.last_trade_time).toLocaleString() : 'N/A'}</p>
                            <p><strong>Total Allocation:</strong> {(totalAllocation * 100).toFixed(2)}%</p>
                          </div>
                          <table className="snapshot-table">
                            <thead>
                              <tr>
                                <th>Symbol</th>
                                <th>Allocation (%)</th>
                                <th>Price (USDT)</th>
                                <th>Quantity</th>
                                <th>Position Value</th>
                                <th>Average Price</th>
                              </tr>
                            </thead>
                            <tbody>
                              {symbols.map(([symbol, data]) => (
                                <tr key={symbol}>
                                  <td>{symbol}</td>
                                  <td>{(data.allocation * 100).toFixed(2)}%</td>
                                  <td>${data.current_price.toFixed(2)}</td>
                                  <td>{data.quantity.toFixed(6)}</td>
                                  <td>${data.position_value.toFixed(2)}</td>
                                  <td>${data.average_price.toFixed(2)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      );
                    })}
                  </div>
                </>
              ) : (
                <div className="no-data-message">No snapshot data available</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BinanceChart;
