function data_read = read_and_filter_rul_data(filepath, default_spd, default_trq, default_pwr, default_eff)
% 讀取馬達 RUL 資料並進行電壓/電流的 FFT 與 IIR 濾波處理
% 支援 parquet/csv 檔案
% 
% 輸入參數：
% filepath      : 檔案路徑 (.parquet 或 .csv)
% default_spd   : 預設速度 (若 CSV 無 Speed 欄位)
% default_trq   : 預設轉矩
% default_pwr   : 預設功率
% default_eff   : 預設效率
%
% 輸出為 data_read 結構體

    arguments
        filepath (1,:) char
        default_spd double = 0
        default_trq double = 0
        default_pwr double = 0
        default_eff double = 0
    end

    data_read = struct();

    if isfile(filepath)
        [~,~,ext] = fileparts(filepath);
        switch ext
            case '.parquet'
                df = readtable(filepath, 'FileType', 'parquet');

                data_read.UnixTime   = df.("Unix Time")(1);
                data_read.Speed      = df.Speed(1);
                data_read.Torque     = df.Torque(1);
                data_read.Power      = df.Power(1);
                data_read.Efficiency = df.Efficiency(1);

                data_read.VoltageAlpha = df.("Voltage alpha"){1}(:);
                data_read.VoltageBeta  = df.("Voltage beta"){1}(:);
                data_read.CurrentAlpha = df.("Current alpha"){1}(:);
                data_read.CurrentBeta  = df.("Current beta"){1}(:);

                if ismember("raw_pico_data", df.Properties.VariableNames)
                    raw_pico_data = df.raw_pico_data{1};
                    data_read.VibrationData = raw_pico_data(:);
                    data_read.VibrationRMS = sqrt(mean(data_read.VibrationData.^2));
                end

                unix_time = data_read.UnixTime;
                if unix_time >= 1748361600 && unix_time <= 1750780800
                    disp("Current data divided by 10 due to incorrect scaling.");
                    data_read.CurrentAlpha = data_read.CurrentAlpha / 10;
                    data_read.CurrentBeta  = data_read.CurrentBeta  / 10;
                end

                % 濾波降採樣
                [data_read.VoltageAlphaDownsample, ~] = filter_top_n_frequencies(data_read.VoltageAlpha, 5);
                [data_read.VoltageBetaDownsample,  ~] = filter_top_n_frequencies(data_read.VoltageBeta, 5);
                data_read.CurrentAlphaDownsample = lowpass_filter(data_read.CurrentAlpha, 2000, 20000, 2);
                data_read.CurrentBetaDownsample  = lowpass_filter(data_read.CurrentBeta, 2000, 20000, 2);

            case '.csv'
                fid = fopen(filepath); first_line = fgetl(fid); fclose(fid);
                tokens = strsplit(first_line, ',');
                unix_time = str2double(tokens{2});

                df = readtable(filepath, 'HeaderLines', 1);
                data_read.UnixTime   = unix_time;
                data_read.Speed      = default_spd;
                data_read.Torque     = default_trq;
                data_read.Power      = default_pwr;
                data_read.Efficiency = default_eff;

                data_read.VoltageAlpha = df.V_alpha(:);
                data_read.VoltageBeta  = df.V_beta(:);
                data_read.CurrentAlpha = df.I_alpha(:);
                data_read.CurrentBeta  = df.I_beta(:);

                % 可加入濾波（若有需要）
            otherwise
                disp(['Unsupported file format: ' filepath]);
        end
    else
        disp(['檔案不存在：' filepath]);
    end
end

%% 子函式：FFT保留前 n 頻率分量
function [filtered_signal, filtered_fft] = filter_top_n_frequencies(signal, n)
    signal = signal(:);
    N = length(signal);
    fft_vals = fft(signal);
    fft_mags = abs(fft_vals);

    [~, idx_sorted] = sort(fft_mags(2:floor(N/2)), 'descend');
    indices = idx_sorted(1:n) + 1;

    mask = false(N,1);
    mask(1) = true; 
    mask(indices) = true;
    mask(end - indices + 2) = true;

    filtered_fft = zeros(N,1);
    filtered_fft(mask) = fft_vals(mask);
    filtered_signal = real(ifft(filtered_fft));
end

%% 子函式：低通濾波器 (Butterworth)
function y = lowpass_filter(data, cutoff, fs, order)
    data = data(:);
    nyq = 0.5 * fs;
    normal_cutoff = cutoff / nyq;
    [b, a] = butter(order, normal_cutoff, 'low');
    y = filtfilt(b, a, data);
end
