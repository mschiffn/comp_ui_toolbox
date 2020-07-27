%
% abstract superclass for all peak widths
%
% author: Martin F. Schiffner
% date: 2020-07-06
% modified: 2020-07-06
%
classdef (Abstract) width < processing.metrics.metric

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        interval ( 1, 1 ) math.interval { mustBeNonempty } = math.interval      % interval to be inspected
        boundary_dB ( 1, 1 ) double { mustBeNegative, mustBeNonempty } = -6     % boundary value in dB

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = width( intervals, boundaries_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure zero to two arguments
            narginchk( 0, 2 );

            %
            if nargin < 1 || isempty( intervals )
                intervals = math.interval( -Inf, Inf )
            end

            % property validation functions ensure class math.interval for intervals

            % ensure nonempty boundaries_dB
            if nargin < 2 || isempty( boundaries_dB )
                boundaries_dB = -6;
            end

            % property validation functions ensure nonempty negative double for boundaries_dB

            % ensure equal number of dimensions and sizes
            [ intervals, boundaries_dB ] = auxiliary.ensureEqualSize( intervals, boundaries_dB );

            %--------------------------------------------------------------
            % 2.) create widths
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.metric( size( intervals ) );

            % iterate widths
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).interval = intervals( index_object );
                objects( index_object ).boundary_dB = boundaries_dB( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = width( ROIs, boundaries_dB )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate metric (scalar)
        %------------------------------------------------------------------
        function results = evaluate_scalar( width, signal_matrix )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.metrics.metric (scalar) for width

            % ensure class math.sequence_increasing_regular
            if ~isa( signal_matrix.axis, 'math.sequence_increasing_regular' )
                errorStruct.message = sprintf( 'signal_matrix.axis must be math.sequence_increasing_regular!', index_signal );
                errorStruct.identifier = 'evaluate_scalar:NoRegularAxis';
                error( errorStruct );
            end

            % calling function ensures class processing.signal_matrix (scalar) for signal_matrix

            %--------------------------------------------------------------
            % 2.) compute width metric (scalar)
            %--------------------------------------------------------------
            % specify cell array for results
            results = cell( 1, signal_matrix.N_signals );

            % detect valid interval points
            indicator_interval = iselement( width.interval, signal_matrix.axis );

            % iterate signals
            for index_signal = 1:signal_matrix.N_signals

                % logarithmic compression
                samples_dB = illustration.dB( signal_matrix.samples( indicator_interval, index_signal ), 20 );

                % peak detection
                [ peaks, peaks_indices ] = findpeaks( samples_dB, 'MINPEAKHEIGHT', - eps( 0 ) );
                N_peaks = numel( peaks_indices );

                % ensure single peak
                if N_peaks ~= 1
                    errorStruct.message = sprintf( 'Signal %d does not have a single peak!', index_signal );
                    errorStruct.identifier = 'evaluate_scalar:NoSinglePeak';
                    error( errorStruct );
                end

                % initialize peak widths w/ zeros
                widths_out{ index_object }{ index_signal } = repmat( delta, size( peaks ) );

                % find width of peaks
                for index_peak = 1:N_peaks

                    index_lb = peaks_indices( index_peak );
                    index_ub = peaks_indices( index_peak );

                    % lower bound
                    while samples_dB( index_lb ) >= peaks( index_peak ) + thresholds_dB( index_object )

                        index_lb = index_lb - 1;
                    end
                    index_lb = index_lb + 1;
    
                    % upper bound
                    while samples_dB( index_ub ) >= peaks( index_peak ) + thresholds_dB( index_object )

                        index_ub = index_ub + 1;
                    end
                    index_ub = index_ub - 1;

                    % compute extent of peak
                    widths_out{ index_object }{ index_signal }( index_peak ) = ( index_ub - index_lb + 1 ) * delta;

                end % for index_peak = 1:N_peaks

            end % for index_signal = 1:signal_matrix.N_signals

            % concatenate horizontally
            results = cat( 2, results{ : } );

        end % function results = evaluate_scalar( width, signal_matrix )

	end % methods (Access = protected, Hidden)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (abstract, protected, and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate metric (samples)
        %------------------------------------------------------------------
        result = evaluate_samples( width, delta_V, indicator )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) width < processing.metrics.metric
