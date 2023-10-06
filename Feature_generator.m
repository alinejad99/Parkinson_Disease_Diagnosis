%% Loading and ...
clc; clear; close all;
PD = load('PD.mat').PD;       % 60 samples for 14 PDs
CTL = load('CTL.mat').Control;    % 61 samples for 14 CTLs
%%
% % deleteing first two samples
% %PD = PD;     % 60 samples for 14 PDs = 840
% CTL = CTL( :, :, 15:854);    % 60 samples for 14 CTLs = 840
data_concatenated = cat(3, PD, CTL);

samples = size(data_concatenated,3); %samples number
cn = 63; %channels number
sampling_rate = 100;
% stat_features = zeros(samples, 10);  %manually calculated
% freq_features = zeros(samples, 10); %manually calculated

%%
%features_1 = FeatureExtractor(data_concatenated, fs);
features = zeros(samples,5544);
for sample = 1:samples
    data = data_concatenated(:, :, sample);

    variance_feat = variance(data);
    correlation_feat = channel_correlation(data);
    kurtosis_feat = kurtosis_data(data);
    skewness_feat = skewness_data(data);
    histogram_feat = histogram_channels(data);
    ar_coefficients = ar_model_coefficients(data,4);
    entropy_feat = shannon_entropy(data);
    max_freq_feat = maximum_frequency(data, sampling_rate);
    mean_freqs = mean_median_frequency(data, sampling_rate);
    power_ratio_feat = power_spectral_ratio(data, sampling_rate);
    
    feature = [variance_feat,correlation_feat, kurtosis_feat,skewness_feat,histogram_feat,ar_coefficients,entropy_feat,max_freq_feat,mean_freqs,power_ratio_feat];
    


    % Assuming your array is called 'data'
    mean_amplitudes = mean(abs(feature), 1);

    % Calculate the standard deviation amplitude of each column
    std_amplitudes = std(abs(feature), 0, 1);
    std_amplitudes(std_amplitudes == 0) = 1; % Replace zeros with 1

    % Divide each element in the column by the standard deviation amplitude and subtract the mean amplitude
    normalized_data = (abs(feature) - mean_amplitudes) ./ std_amplitudes;
    



    features_flat = reshape(normalized_data, 1, []);
    features(sample,:) = features_flat;

end
%%

% Assuming your array is called 'data'
[row_indices, col_indices] = find(isnan(features));
nan_locations = [row_indices, col_indices];

num_nans = sum(isnan(features(:)));
%%
save('all_features.mat','features');
%%
testing_nan = maximum_frequency(data_concatenated(:,:,115), sampling_rate);


%% Functions
function variances = variance(data)
    variances = var(data, 0, 2);
end

function correlation = channel_correlation(data)
    [num_channels, ~, ~] = size(data);
    reshaped_data = reshape(data, num_channels, []);
    reshaped_data = reshaped_data.';
    correlation = corrcoef(reshaped_data);
end

function kurtosis_values = kurtosis_data(data)
    kurtosis_values = kurtosis(data, 0, 2);
end

function skewness_values = skewness_data(data)
    skewness_values = skewness(data, 0, 2);
end

function histograms = histogram_channels(data)
    histograms = zeros(size(data, 1), 10);  % 10 bins for the histogram
    for i = 1:size(data, 1)
        histograms(i, :) = histcounts(data(i, :), 'BinMethod', 'auto', 'NumBins', 10);
    end
end

function coefficients = ar_model_coefficients(data, order)
    num_channels = size(data, 1);
    coefficients = zeros(num_channels, order);
    
    for i = 1:num_channels
        time_series = double(data(i, :));
        iddata_obj = iddata(time_series', [], 1);
        model = ar(iddata_obj, order);
        coefficients(i, :) = model.A(2:end)';
    end
end


function entropies = shannon_entropy(data)
    entropies = -sum(data .* log2(data), 2);
end

function max_freqs = maximum_frequency(data, sampling_rate)
    N = size(data, 2);
    freqs = (0:N/2-1) * (sampling_rate / N);
    magnitudes = abs(fft(data, [], 2));
    [~, max_freq_indices] = max(magnitudes(:, 1:N/2), [], 2);
    max_freq = freqs(max_freq_indices);
    max_freqs = max_freq';
end

function avg_freqs = mean_median_frequency(data, sampling_rate)
    N = size(data, 2);
    freqs = (0:N/2-1) * (sampling_rate / N);
    magnitudes = abs(fft(data, [], 2));
    avg_freqs = sum(freqs .* magnitudes(:, 1:N/2), 2) ./ sum(magnitudes(:, 1:N/2), 2);
    %median_freqs = median(freqs, 2);
end

function power_ratios = power_spectral_ratio(data, sampling_rate)
    N = size(data, 2);
    freqs = (0:N/2-1) * (sampling_rate / N);
    magnitudes = abs(fft(data, [], 2));

    delta_band_mask = freqs >= 0.5 & freqs <= 4;
    theta_band_mask = freqs > 4 & freqs <= 8;
    alpha_band_mask = freqs > 8 & freqs <= 13;
    beta_band_mask = freqs > 13 & freqs <= 30;
    gamma_band_mask = freqs > 30 & freqs <= 60;

    delta_power = sum(magnitudes(:, delta_band_mask), 2);
    theta_power = sum(magnitudes(:, theta_band_mask), 2);
    alpha_power = sum(magnitudes(:, alpha_band_mask), 2);
    beta_power = sum(magnitudes(:, beta_band_mask), 2);
    gamma_power = sum(magnitudes(:, gamma_band_mask), 2);
    total_power = sum(magnitudes(:, 1:N/2), 2);

    power_ratios = [delta_power, theta_power, alpha_power, beta_power, gamma_power] ./ total_power;
end

%% Feature extracting
% for i = 1:samples
%     data = data_concatenated(:,:,i);
%     
%     % Variance
%     var_data = var(data,0,2);
%     stat_features(i,1:cn) = var_data;
% 
%     % Correlation 
%     corr_data = corrcoef(data');
%     m = 1;
%     for j = 1:cn
%         for k = j+1:cn
%             stat_features(i,cn+m) = corr_data(k,j);
%             m = m+1;
%         end
%     end
%     
%     % Kurtosis
%     kurtosis_data = kurtosis(data,1,2);
%     stat_features(i,(cn+nchoosek(cn,2)+1):(2*cn+nchoosek(cn,2))) = kurtosis_data;
% 
%     % Skewness
%     skewness_data = skewness(data,1,2);
%     stat_features(i,(2*cn+nchoosek(cn,2))+1:(3*cn+nchoosek(cn,2))) = skewness_data;
% 
%     % Histogram of 6 Channels
%     num_bins = 20;
%     for j = 1:cn
%         time_series = data(j,:);
%         [N , edges] = histcounts(time_series,num_bins);
%         %idx = (3*cn+nchoosek(cn,2)+1)+(j-1)*num_bins
%         stat_features(i,((3*cn+nchoosek(cn,2)+1)+(j-1)*num_bins:(3*cn+nchoosek(cn,2))+(j)*num_bins)) = N;
%     end
%     
%     % AR Model Coefficients
%         for j = 1:cn
%             time_series = data_concatenated(j, :);
%             ar_time_series = aryule(time_series, 10);
%             stat_features(i, (((3 * cn + nchoosek(cn, 2)) + (cn) * num_bins + 1) + (j - 1) * 10):((3 * cn + nchoosek(cn, 2)) + (cn) * num_bins + (j) * 10)) = ar_time_series(2:11);
%         end
% 
%     
%     % Shannon Entropy
%     
%     for j = 1:cn
%         time_series = data(j,:);
%         shannon_entropy =  wentropy(time_series,'shannon');
%         stat_features(i,(3*cn+nchoosek(cn,2))+(cn)*(num_bins+10)+1:(4*cn+nchoosek(cn,2))+(cn)*(num_bins+10)) = shannon_entropy;
%     end
%     
%     % Frequency Domain Features
% 
%     % Calculating the Frequency and Time Vectors
%     t = 0:1/fs:4-(1/fs);
%     df = fs/(length(t)-1);
%     f = -fs/2:df:fs/2;
% 
%     % Max Freq
%     max_freq = zeros(1,cn);
%     for j = 1:cn
%         [I, max_freq(j)] = max(abs(fftshift(fft(data(j,:))*1/fs)));
%     end
%     freq_features(i,1:cn) = abs(f(max_freq));
% 
%     % Mean and Median Freq
%     mean_freq = zeros(1,cn);
%     med_freq = zeros(1,cn);
%     for j = 1:cn
%         mean_freq(j) = meanfreq(data(j,:),fs);
%         med_freq(j) = medfreq(data(j,:),fs);
%     end
%     freq_features(i,cn+1:2*cn) = mean_freq;
%     freq_features(i,2*cn+1:3*cn) = med_freq;
%     
%     % Power Spectral Ratio
%     
%     delta_range = [1 3];
%     theta_range = [4 7];
%     alpha_range = [8 11];
%     beta_range  = [12 30];
%     gamma_range = [31 60];
%     total_range = [1 60];
%     
%     frequency_range = [delta_range; theta_range; alpha_range; beta_range; gamma_range];
% 
%     for j = 1:cn
%         for k = 1:5
%         Range = frequency_range(k,:);
%         time_series = data(j,:);
%         total_power = bandpower(time_series,fs,total_range);
%         each_band_power = bandpower(time_series,fs,Range);
%         freq_features(i,(j-1)*5+((3*cn)+k)) = each_band_power/total_power;
%         end
%     end
%     
%     % Gathering the whole features
%     all_features = [stat_features freq_features];
%    
% end
% 
% 
% 
% 

%

% num_bins = 20;
% num_samples = size(data_conc, 3);
% stat_features = zeros(num_samples, 3*cn + nchoosek(cn,2) + cn*(num_bins+10) + cn*10 + cn);
% freq_features = zeros(num_samples, 3*cn);
% 
% % Calculating the Frequency and Time Vectors
% t = 0:1/fs:4-(1/fs);
% df = fs/(length(t)-1);
% % f = -fs/2:df:fs/2;
% fpass = [0.5 60];
% % Calculate the frequency vector
% N = size(data_conc, 2); % Length of the time series
% f = (-fs/2) + (0:N-1) * (fs/N);
% 
% % Calculate the normalized frequency range
% normalized_fpass = fpass / (fs/2);
% 
% for i = 1:num_samples
% 
%     u_data = data_conc(:, :, i);
%     data = zeros(size(u_data));
%     for j = 1:cn
%         data(j, :) = bandpass(u_data(j, :), normalized_fpass, fs);
%     end
%     var_data = var(data, 0, 2);
%     stat_features(i, 1:cn) = var_data;
% 
%     % Correlation
%     corr_data = corrcoef(data');
%     m = 1;
%     for j = 1:cn
%         for k = j+1:cn
%             stat_features(i, cn+m) = corr_data(k, j);
%             m = m + 1;
%         end
%     end
%     
%     % Kurtosis
%     kurtosis_data = kurtosis(data, 1, 2);
%     stat_features(i, (cn + 1):(2*cn)) = kurtosis_data;
% 
%     % Skewness
%     skewness_data = skewness(data, 1, 2);
%     stat_features(i, (2*cn + 1):(3*cn)) = skewness_data;
% 
%     % Histogram of 6 Channels
%     num_bins = 20;
%     for j = 1:cn
%         time_series = data(j, :);
%         N = histcounts(time_series, num_bins);
%         stat_features(i, (3*cn + nchoosek(cn, 2) + 1) + (j-1)*num_bins : (3*cn + nchoosek(cn, 2)) + j*num_bins) = N;
%     end
%     
%     % AR Model Coefficients
%     for j = 1:cn
%         time_series = data(j, :);
%         ar_time_series = aryule(time_series, 10);
%         stat_features(i, ((3*cn + nchoosek(cn, 2)) + (cn) * num_bins + 1) + (j - 1)*10 : ((3*cn + nchoosek(cn, 2)) + (cn) * num_bins + j*10)) = ar_time_series(2:11);
%     end
%     
%     % Shannon Entropy
%     for j = 1:cn
%         time_series = data(j, :);
%         shannon_entropy = wentropy(time_series, 'shannon');
%         stat_features(i, (3*cn + nchoosek(cn, 2)) + (cn)*(num_bins+10) + 1 : (4*cn + nchoosek(cn, 2)) + (cn)*(num_bins+10)) = shannon_entropy;
%     end
%     
%     % Frequency Domain Features
%         % Power Spectral Ratio
%     for j = 1:cn
%         time_series = data(j, :);
% 
%         % Max Freq
%         [I, max_freq] = max(abs(fftshift(fft(time_series)*1/fs)));
%         freq_features(i, j) = abs(f(max_freq));
% 
%         % Mean and Median Freq
%         mean_freq = meanfreq(time_series, fs);
%         med_freq = medfreq(time_series, fs);
%         freq_features(i, cn+j) = mean_freq;
%         freq_features(i, 2*cn+j) = med_freq;
% 
%         % Power Spectral Ratio
%         delta_range = [1 3];
%         theta_range = [4 7];
%         alpha_range = [8 11];
%         beta_range  = [12 30];
%         gamma_range = [31 60];
%         total_range = [1 60];
% 
%         frequency_ranges = [delta_range; theta_range; alpha_range; beta_range; gamma_range];
% 
%         for k = 1:5
%             range = frequency_ranges(k,:);
%             time_series = data(j,:);
%             % Apply bandpass filter
%             filtered_data = bandpass(time_series, range, fs);
% 
%             % Calculate power using modified frequency vector
%             freq_indices = (f >= range(1)) & (f <= range(2));
%             total_power = bandpower(filtered_data, fs, total_range, 'psd');
%             each_band_power = bandpower(filtered_data, fs, range, 'psd');
%             freq_features(i, (j-1)*5 + ((3*cn) + k)) = each_band_power / total_power;
%         end
%     end
%     % Gathering the whole features
%     all_features = [stat_features freq_features];
% end
% 
% 
% % Save as PD.mat
% save('PD.mat', 'all_features');
% 
% 
% % y_labels = zeros(n, 1);
% % y_labels(1:100) = ones(100, 1)
% % y_labels = 
% % features_best = fscnca(all_features,
% % save('features_1.mat', 'all_features');
% 
% 
% 
% 
% 
% 
% %%
% % clc; clear; close all;
% % load('data_Train_TrainLabel_Test.mat');
% % load('features.mat');
% 
% % % Normalizing
% % [features_norm_train, PS] = mapstd(all_features(1:90,:)',0,1);
% % features_norm_train = features_norm_train';
% % features_norm_test = mapstd('apply',all_features(91:180,:)',PS);
% % features_norm_test = features_norm_test';
% % 
% % % Univariate feature ranking for classification using chi-square tests
% % [ranking_chi, score_chi] = fscchi2(features_norm_train,EEG_train_label');
% % 
% % % PCA Calculating of the features
% % loadings_train = pca(features_norm_train);
% % top_train = features_norm_train*loadings_train(:,1:5);
% % loadings_test = pca(features_norm_test);
% % top_test = features_norm_test*loadings_test(:,1:5);
% % 
% % % Creating the Training Target
% % Y = zeros(1,90) ;
% % Y(1,EEG_train_label==1) = 1 ;
% % Y(1,EEG_train_label==-1) = 0 ;
% 

