%
% superclass for all signal matrices
%
% author: Martin F. Schiffner
% date: 2019-03-27
% modified: 2020-01-16
%
classdef signal_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = protected)

        % independent properties
        axis ( 1, 1 ) math.sequence_increasing
        samples ( :, : )

        % dependent properties
        N_signals ( 1, 1 ) { mustBeInteger, mustBePositive } = 1

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = signal_matrix( axes, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for zero arguments
            if nargin == 0
                return;
            end

            % ensure class math.sequence_increasing
            if ~isa( axes, 'math.sequence_increasing' )
                errorStruct.message = 'axes must be math.sequence_increasing!';
                errorStruct.identifier = 'signal_matrix:NoIncreasingSequence';
                error( errorStruct );
            end

            % ensure cell array for samples
            if ~iscell( samples )
                samples = { samples };
            end

            % multiple axes / single samples
            if ~isscalar( axes ) && isscalar( samples )
                samples = repmat( samples, size( axes ) );
            end

            % single axes / multiple samples
            if isscalar( axes ) && ~isscalar( samples )
                axes = repmat( axes, size( samples ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( axes, samples );

            %--------------------------------------------------------------
            % 2.) create signal matrices
            %--------------------------------------------------------------
            % numbers of axis members
            N_members = abs( axes );

            % repeat default signal matrix
            objects = repmat( objects, size( axes ) );

            % iterate signal matrices
            for index_object = 1:numel( objects )

                % ensure matrix for samples{ index_object }
                if ~ismatrix( samples{ index_object } )
                    errorStruct.message = sprintf( 'samples{ %d } must be a matrix!', index_object );
                    errorStruct.identifier = 'signal_matrix:NoMatrix';
                    error( errorStruct );
                end

                % ensure numeric type or physical_values.physical_quantity for samples{ index_object }
                if ~( isnumeric( samples{ index_object } ) || isa( samples{ index_object }, 'physical_values.physical_quantity' ) )
                    errorStruct.message = sprintf( 'samples{ %d } must be numeric or physical_values.physical_quantity!', index_object );
                    errorStruct.identifier = 'signal_matrix:NoNumericOrPhysicalQuantity';
                    error( errorStruct );
                end

                % ensure correct sizes
                if N_members( index_object ) ~= size( samples{ index_object }, 1 )
                    errorStruct.message = sprintf( 'Cardinality of axes( %d ) must match the size of samples{ %d } along the first dimension!', index_object, index_object );
                    errorStruct.identifier = 'signal_matrix:SizeMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).axis = axes( index_object );
                objects( index_object ).samples = samples{ index_object };

                % set dependent properties
                objects( index_object ).N_signals = size( samples{ index_object }, 2 );

            end % for index_object = 1:numel( objects )

        end % function objects = signal_matrix( axes, samples )

        %------------------------------------------------------------------
        % orthonormal discrete Fourier transform (DFT)
        %------------------------------------------------------------------
        function [ signal_matrices, N_dft, deltas ] = DFT( signal_matrices, varargin )
% TODO: generalize for complex-valued samples
% TODO: circshift not suitable for Fourier transform?!

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix
            if ~isa( signal_matrices, 'processing.signal_matrix' )
                errorStruct.message = 'signal_matrices must be processing.signal_matrix!';
                errorStruct.identifier = 'DFT:NoSignalMatrices';
                error( errorStruct );
            end

            % ensure real-valued samples
            indicator = cellfun( @( x ) ~isreal( x ), { signal_matrices.samples } );
            if any( indicator( : ) )
                errorStruct.message = 'signal_matrices.samples must be real-valued!';
                errorStruct.identifier = 'DFT:NoRealSamples';
                error( errorStruct );
            end

            % ensure equal subclasses of math.sequence_increasing_regular_quantized
            auxiliary.mustBeEqualSubclasses( 'math.sequence_increasing_regular_quantized', signal_matrices.axis );

            % extract regular axes
            axes_t = reshape( [ signal_matrices.axis ], size( signal_matrices ) );

            % ensure equal subclasses of physical_values.time
% TODO: generalize for arbitrary physical units
% ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.time', axes_t.delta );

            % extract integer bounds (int64) and temporal sampling periods
            lbs_q_signal = reshape( double( [ axes_t.q_lb ] ), size( signal_matrices ) );
            ubs_q_signal = reshape( double( [ axes_t.q_ub ] ), size( signal_matrices ) );
            deltas = reshape( [ axes_t.delta ], size( signal_matrices ) );

            % ensure nonempty T_ref
            if nargin >= 2 && ~isempty( varargin{ 1 } )
                T_ref = varargin{ 1 };
            else
                T_ref = ( ubs_q_signal - lbs_q_signal + 1 ) .* deltas;
            end

            % ensure class physical_values.time
% TODO: generalize for arbitrary physical units
% ensure equal subclasses of class( deltas )
            if ~isa( T_ref, 'physical_values.time' )
                errorStruct.message = 'T_ref must be physical_values.time!';
                errorStruct.identifier = 'DFT:NoTimes';
                error( errorStruct );
            end

            % ensure nonempty intervals_f
            if nargin >= 3 && ~isempty( varargin{ 2 } )
                intervals_f = varargin{ 2 };
            else
                intervals_f = math.interval( physical_values.hertz( zeros( size( deltas ) ) ), 1 ./ ( 2 * deltas ) );
            end

            % ensure class math.interval
            if ~isa( intervals_f, 'math.interval' )
                errorStruct.message = 'intervals_f must be math.interval!';
                errorStruct.identifier = 'DFT:NoIntervals';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.frequency
% TODO: generalize for arbitrary physical units
% ensure equal subclasses of reciprocal( class( deltas ) )
            auxiliary.mustBeEqualSubclasses( 'physical_values.frequency', intervals_f.lb );

            % multiple signal_matrices / single T_ref
            if ~isscalar( signal_matrices ) && isscalar( T_ref )
                T_ref = repmat( T_ref, size( signal_matrices ) );
            end

            % multiple signal_matrices / single intervals_f
            if ~isscalar( signal_matrices ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( signal_matrices ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_matrices, T_ref, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute orthonormal discrete Fourier transforms
            %--------------------------------------------------------------
            % numbers of temporal samples
            N_samples_signal = abs( axes_t );

            % compute numbers of points in the DFTs (numbers of intervals)
            N_dft = round( T_ref ./ deltas );
            if any( abs( T_ref ./ deltas - N_dft ) > eps( N_dft ) )
                errorStruct.message = sprintf( 'T_ref must be integer multiples of deltas!' );
                errorStruct.identifier = 'DFT:InvalidTRef';
                error( errorStruct );
            end

            % numbers of zeros to pad
            N_zeros_pad = N_dft - N_samples_signal;

            % ensure that numbers of samples do not exceed the order of the DFT
            if any( N_zeros_pad( : ) < 0 )
                errorStruct.message = sprintf( 'Numbers of signal samples exceed orders of the DFTs!' );
                errorStruct.identifier = 'DFT:ShortTRec';
                error( errorStruct );
            end

            % compute axes of relevant frequencies
            axes_f = discretize( intervals_f, 1 ./ T_ref );
            lbs_q_f = reshape( [ axes_f.q_lb ], size( axes_f ) );
            ubs_q_f = reshape( [ axes_f.q_ub ], size( axes_f ) );

            % ensure Nyquist criterion
% TODO: include critical Nyquist frequency 1 ./ ( 2 * deltas )? => real-valued Fourier coefficient
            if any( lbs_q_f( : ) < 0 ) || any( ubs_q_f( : ) > floor( N_dft( : ) / 2 ) )
                errorStruct.message = 'Invalid intervals_f!';
                errorStruct.identifier = 'DFT:InvalidFrequencyIntervals';
                error( errorStruct );
            end

            % iterate signal matrices
            for index_matrix = 1:numel( signal_matrices )

                % specify relevant frequency indices
                indices_relevant = double( lbs_q_f( index_matrix ):ubs_q_f( index_matrix ) ) + 1;

                % zero-pad and shift samples
                samples_act = [ signal_matrices( index_matrix ).samples; zeros( N_zeros_pad( index_matrix ), signal_matrices( index_matrix ).N_signals ) ];
                samples_act = circshift( samples_act, lbs_q_signal( index_matrix ), 1 );

                % compute and truncate DFT
                DFT_act = fft( samples_act, [], 1 ) / sqrt( N_dft( index_matrix ) );
                signal_matrices( index_matrix ).samples = DFT_act( indices_relevant, : );

                % update axis
                signal_matrices( index_matrix ).axis = axes_f( index_matrix );

            end % for index_matrix = 1:numel( signal_matrices )

        end % function [ signal_matrices, N_dft, deltas ] = DFT( signal_matrices, varargin )

        %------------------------------------------------------------------
        % Fourier transform (cf. book:Briggs1995, pp. 40, 41)
        %------------------------------------------------------------------
        function signal_matrices = fourier_transform( signal_matrices, varargin )
% TODO: Fourier transform does not exhibit periodicity!
            %--------------------------------------------------------------
            % 1.) compute orthonormal discrete Fourier transforms (DFTs)
            %--------------------------------------------------------------
            [ signal_matrices, N_dft, deltas ] = DFT( signal_matrices, varargin{ : } );

            %--------------------------------------------------------------
            % 2.) compute Fourier transform samples
            %--------------------------------------------------------------
            for index_matrix = 1:numel( signal_matrices )

                % scale samples
                signal_matrices( index_matrix ).samples = deltas( index_matrix ) * sqrt( N_dft( index_matrix ) ) * signal_matrices( index_matrix ).samples;

            end % for index_matrix = 1:numel( signal_matrices )

        end % function signal_matrices = fourier_transform( signal_matrices, varargin )

        %------------------------------------------------------------------
        % Fourier coefficients (cf. book:Briggs1995, p. 40)
        %------------------------------------------------------------------
        function signal_matrices = fourier_coefficients( signal_matrices, varargin )

            %--------------------------------------------------------------
            % 1.) compute orthonormal discrete Fourier transforms (DFTs)
            %--------------------------------------------------------------
            [ signal_matrices, N_dft ] = DFT( signal_matrices, varargin{ : } );

            %--------------------------------------------------------------
            % 2.) compute Fourier series coefficients
            %--------------------------------------------------------------
            for index_matrix = 1:numel( signal_matrices )

                % scale samples
                signal_matrices( index_matrix ).samples = signal_matrices( index_matrix ).samples / sqrt( N_dft( index_matrix ) );

            end % for index_matrix = 1:numel( signal_matrices )

        end % function signal_matrices = fourier_coefficients( signal_matrices, varargin )

        %------------------------------------------------------------------
        % time-domain signal
        %------------------------------------------------------------------
        function [ signal_matrices, N_samples_t ] = signal( signal_matrices, varargin )
% TODO: transform in place!

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix
            if ~isa( signal_matrices, 'processing.signal_matrix' )
                errorStruct.message = 'signal_matrices must be processing.signal_matrix!';
                errorStruct.identifier = 'signal:NoSignalMatrices';
                error( errorStruct );
            end

            % ensure equal subclasses of math.sequence_increasing_regular_quantized
            auxiliary.mustBeEqualSubclasses( 'math.sequence_increasing_regular_quantized', signal_matrices.axis );

            % extract regular axes
            axes_f = reshape( [ signal_matrices.axis ], size( signal_matrices ) );

            % ensure equal subclasses of physical_values.frequency
            auxiliary.mustBeEqualSubclasses( 'physical_values.frequency', axes_f.delta );

            % extract integer bounds and spectral sampling periods
            lbs_q_signal = reshape( [ axes_f.q_lb ], size( signal_matrices ) );
            ubs_q_signal = reshape( [ axes_f.q_ub ], size( signal_matrices ) );
            deltas = reshape( [ axes_f.delta ], size( signal_matrices ) );

            % ensure nonempty lbs_q
            if nargin >= 2 && ~isempty( varargin{ 1 } )
                lbs_q = varargin{ 1 };
            else
                lbs_q = zeros( size( signal_matrices ) );
            end

            % ensure integer bounds
            mustBeInteger( lbs_q );

            % ensure nonempty delta
            if nargin >= 3 && ~isempty( varargin{ 2 } )
                delta = varargin{ 2 };
            else
                delta = 1 ./ ( 2 * ubs_q_signal .* deltas );
            end

            % ensure class physical_values.time
            if ~isa( delta, 'physical_values.time' )
                errorStruct.message = 'delta must be physical_values.time!';
                errorStruct.identifier = 'signal:NoTime';
                error( errorStruct );
            end

            % multiple signal_matrices / single lbs_q
            if ~isscalar( signal_matrices ) && isscalar( lbs_q )
                lbs_q = repmat( lbs_q, size( signal_matrices ) );
            end

            % multiple signal_matrices / single delta
            if ~isscalar( signal_matrices ) && isscalar( delta )
                delta = repmat( delta, size( signal_matrices ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_matrices, lbs_q, delta );

            %--------------------------------------------------------------
            % 2.) compute time-domain signals
            %--------------------------------------------------------------
            % numbers of spectral samples
            N_samples_f = abs( axes_f );

            % compute time axes
% TODO: noninteger?
% TODO: bandwidth? index_shift >= N_samples_f => ceil( T_rec ./ ( 2 delta ) ) >= N_samples_f => 1 ./ ( 2 * deltas * N_samples_f ) >= delta
% % 1 ./ ( 2 * deltas * ( N_samples_f - 1 ) ) >= delta
            T_rec = 1 ./ deltas;
            N_samples_t = floor( T_rec ./ delta );
            axes_t = math.sequence_increasing_regular_quantized( lbs_q, lbs_q + N_samples_t - 1, delta );
            N_samples_f_max = floor( N_samples_t / 2 ) + 1;

            % numbers of zeros to pad
            N_zeros_pad = N_samples_f_max - N_samples_f;

            % ensure that numbers of spectral samples do not exceed the maximum
            if any( N_zeros_pad( : ) < 0 )
                errorStruct.message = sprintf( 'Number of spectral samples exceeds the maximum!' );
                errorStruct.identifier = 'signal:IntervalMismatch';
                error( errorStruct );
            end

            % specify cell array for samples_td
            samples_td = cell( size( signal_matrices ) );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % zero-pad and shift samples
                samples_act = [ signal_matrices( index_object ).samples; zeros( N_zeros_pad( index_object ), signal_matrices( index_object ).N_signals ) ];
                samples_act = circshift( samples_act, lbs_q_signal( index_object ), 1 );

                % compute signal samples
                samples_td{ index_object } = N_samples_t( index_object ) * ifft( samples_act, N_samples_t( index_object ), 1, 'symmetric' );

                % shift samples
                samples_td{ index_object } = circshift( samples_td{ index_object }, - lbs_q( index_object ), 1 );

            end % for index_object = 1:numel( signal_matrices )

            %--------------------------------------------------------------
            % 3.) create signal matrices
            %--------------------------------------------------------------
            signal_matrices = processing.signal_matrix( axes_t, samples_td );

        end % function [ signal_matrices, N_samples_t ] = signal( signal_matrices, varargin )

        %------------------------------------------------------------------
        % inverse Fourier transform
        %------------------------------------------------------------------
        function signal_matrices = inverse_fourier_transform( signal_matrices, varargin )

            %--------------------------------------------------------------
            % 1.) time-domain signal
            %--------------------------------------------------------------
            [ signal_matrices, N_samples_t ] = signal( signal_matrices, varargin{ : } );

            %--------------------------------------------------------------
            % 2.) scale for inverse Fourier transform
            %--------------------------------------------------------------
            % iterate signal matrices
            for index_matrix = 1:numel( signal_matrices )

                % scale samples
                signal_matrices( index_matrix ).samples = signal_matrices( index_matrix ).samples / ( N_samples_t( index_matrix ) * signal_matrices( index_matrix ).axis.delta );

            end % for index_matrix = 1:numel( signal_matrices )

        end % function signal_matrices = inverse_fourier_transform( signal_matrices, varargin )

        %------------------------------------------------------------------
        % interpolate
        %------------------------------------------------------------------
        function signal_matrices = interpolate( signal_matrices, factors_interp )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix
            if ~isa( signal_matrices, 'processing.signal_matrix' )
                errorStruct.message = 'signal_matrices must be processing.signal_matrix!';
                errorStruct.identifier = 'interpolate:NoSignalMatrices';
                error( errorStruct );
            end

            % ensure equal subclasses of math.sequence_increasing_regular
            auxiliary.mustBeEqualSubclasses( 'math.sequence_increasing_regular', signal_matrices.axis );

            % ensure positive integers
            mustBePositive( factors_interp );
            mustBeInteger( factors_interp );

            % multiple signal_matrices / single factors_interp
            if ~isscalar( signal_matrices ) && isscalar( factors_interp )
                factors_interp = repmat( factors_interp, size( signal_matrices ) );
            end

            % single signal_matrices / multiple factors_interp
            if isscalar( signal_matrices ) && ~isscalar( factors_interp )
                signal_matrices = repmat( signal_matrices, size( factors_interp ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_matrices, factors_interp );

            %--------------------------------------------------------------
            % 2.) interpolate signal matrices
            %--------------------------------------------------------------
            % interpolate axes
            axes_int = interpolate( reshape( [ signal_matrices.axis ], size( signal_matrices ) ), factors_interp );

            % numbers of samples
            N_samples_int = abs( axes_int );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                signal_matrices( index_object ).samples = interpft( signal_matrices( index_object ).samples, N_samples_int( index_object ), 1 );
                signal_matrices( index_object ).axis = axes_int( index_object );

            end % for index_object = 1:numel( signal_matrices )

        end % function signal_matrices = interpolate( signal_matrices, factors_interp )

        %------------------------------------------------------------------
        % apply time gain compensation curves
        %------------------------------------------------------------------
        function signal_matrices = apply_tgc( signal_matrices, TGCs )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix for signal_matrices
            if ~isa( signal_matrices, 'processing.signal_matrix' )
                errorStruct.message = 'signal_matrices must be processing.signal_matrix!';
                errorStruct.identifier = 'apply_tgc:NoSignalMatrices';
                error( errorStruct );
            end

            % ensure class regularization.tgc.curve for TGCs
            if ~isa( TGCs, 'regularization.tgc.curve' )
                errorStruct.message = 'TGCs must be regularization.tgc.curve!';
                errorStruct.identifier = 'apply_tgc:NoTGC';
                error( errorStruct );
            end

            % multiple signal_matrices / single TGCs
            if ~isscalar( signal_matrices ) && isscalar( TGCs )
                TGCs = repmat( TGCs, size( signal_matrices ) );
            end

            % single signal_matrices / multiple TGCs
            if isscalar( signal_matrices ) && ~isscalar( TGCs )
                signal_matrices = repmat( signal_matrices, size( TGCs ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_matrices, TGCs );

            %--------------------------------------------------------------
            % 2.) apply time gain compensation curves
            %--------------------------------------------------------------
            % extract axes
            axes = reshape( [ signal_matrices.axis ], size( signal_matrices ) );

            % sample time gain compensation curves
            samples_tgc = sample_curve( TGCs, axes );

            % ensure cell array for samples_tgc
            if ~iscell( samples_tgc )
                samples_tgc = { samples_tgc };
            end

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % apply TGC curve
                signal_matrices( index_object ).samples = signal_matrices( index_object ).samples .* samples_tgc{ index_object };

            end % for index_object = 1:numel( signal_matrices )

        end % function signal_matrices = apply_tgc( signal_matrices, TGCs )

        %------------------------------------------------------------------
        % estimate exponential decays
        %------------------------------------------------------------------
        function [ exponents, factors ] = estimate_decay( signal_matrices )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix for signal_matrices
            if ~isa( signal_matrices, 'processing.signal_matrix' )
                errorStruct.message = 'signal_matrices must be processing.signal_matrix!';
                errorStruct.identifier = 'estimate_decay:NoSignalMatrices';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) estimate exponential decays
            %--------------------------------------------------------------
            % specify cell arrays
            exponents = cell( size( signal_matrices ) );
            factors = cell( size( signal_matrices ) );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % matrix
                X = [ ones( abs( signal_matrices( index_object ).axis ), 1 ), double( signal_matrices( index_object ).axis.members ) ];

                % observations
%                 Y = 20 * log10( double( abs( hilbert( signal_matrices( index_object ).samples ) ) ) );

                Y = illustration.dB( sum( double( signal_matrices( index_object ).samples ).^2, 2 ), 10 );

                % detect useful support
                indicator = Y >= -40;

                % least-squares estimate
                theta = X( indicator, : ) \ Y( indicator, : );

                % extract parameters
                exponents{ index_object } = - log( 10 ) * physical_values.hertz( theta( 2, : ) ) / 20;
                factors{ index_object } = exp( log( 10 ) * theta( 1, : ) / 20 );

            end % for index_object = 1:numel( signal_matrices )

            % avoid cell arrays for single signal_matrices
            if isscalar( signal_matrices )
                exponents = exponents{ 1 };
                factors = factors{ 1 };
            end

        end % function [ exponents, factors ] = estimate_decay( signal_matrices )

        %------------------------------------------------------------------
        % find widths of peaks
        %------------------------------------------------------------------
        function widths_out = widths( signal_matrices, thresholds_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix for signal_matrices
            if ~isa( signal_matrices, 'processing.signal_matrix' )
                errorStruct.message = 'signal_matrices must be processing.signal_matrix!';
                errorStruct.identifier = 'estimate_decay:NoSignalMatrices';
                error( errorStruct );
            end

            % ensure nonempty negative thresholds_dB
            mustBeNegative( thresholds_dB );
            mustBeNonempty( thresholds_dB );

            % multiple signal_matrices / single thresholds_dB
            if ~isscalar( signal_matrices ) && isscalar( thresholds_dB )
                thresholds_dB = repmat( thresholds_dB, size( signal_matrices ) );
            end

            % single signal_matrices / multiple thresholds_dB
            if isscalar( signal_matrices ) && ~isscalar( thresholds_dB )
                signal_matrices = repmat( signal_matrices, size( thresholds_dB ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_matrices, thresholds_dB );

            %--------------------------------------------------------------
            % 2.) compute widths
            %--------------------------------------------------------------
            % specify cell array for widths_out
            widths_out = cell( size( signal_matrices ) );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % specify cell array for widths_out{ index_object }
                widths_out{ index_object } = cell( 1, signal_matrices( index_object ).N_signals );

                % iterate signals
                for index_signal = 1:signal_matrices( index_object ).N_signals

                    %
                    samples_dB = illustration.dB( signal_matrices( index_object ).samples( :, index_signal ), 20 );
                    [ peaks, peaks_indices ] = findpeaks( samples_dB, 'MINPEAKHEIGHT', thresholds_dB( index_object ) / 3 );
                    N_peaks = numel( peaks_indices );

                    % initialize peak widths w/ zeros
                    delta = signal_matrices( index_object ).axis.members( 2 ) - signal_matrices( index_object ).axis.members( 1 );
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

                end % for index_signal = 1:signal_matrices( index_object ).N_signals

%                 N_peaks_per_signal = cellfun( @numel, widths_out{ index_object } );
%                 if all( N_peaks_per_signal == 1 )
%                     widths_out{ index_object } = reshape( cat( 1, widths_out{ index_object }{ : } ), size( widths_out{ index_object } ) );
%                 end

                % avoid cell array for single signal
                if signal_matrices( index_object ).N_signals == 1
                    widths_out{ index_object } = widths_out{ index_object }{ 1 };
                end

            end % for index_object = 1:numel( signal_matrices )

            % avoid cell array for single signal_matrices
            if isscalar( signal_matrices )
                widths_out = widths_out{ 1 };
            end

        end % function widths_out = widths( signal_matrices, thresholds_dB )

        %------------------------------------------------------------------
        % merge compatible signal matrices
        %------------------------------------------------------------------
        function signal_matrix = merge( signal_matrices )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % quick exit for single signal_matrices
            if isscalar( signal_matrices )
                signal_matrix = signal_matrices;
                return;
            end

            % ensure identical axes
            if ~isequal( signal_matrices.axis )
                errorStruct.message = 'All signal matrices must have identical axes!';
                errorStruct.identifier = 'merge:AxesMismatch';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) perform merging
            %--------------------------------------------------------------
            % extract common axis
            axis_mgd = signal_matrices( 1 ).axis;

            % concatenate samples along second dimension
            samples_mgd = { signal_matrices.samples };
            samples_mgd = cat( 2, samples_mgd{ : } );

            %--------------------------------------------------------------
            % 3.) create signal matrix
            %--------------------------------------------------------------
            signal_matrix = processing.signal_matrix( axis_mgd, samples_mgd );

        end % function signal_matrix = merge( signal_matrices )

        %------------------------------------------------------------------
        % element-wise multiplication (overload times method)
        %------------------------------------------------------------------
        function args_1 = times( args_1, args_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure classes processing.signal_matrix
            if ~( isa( args_1, 'processing.signal_matrix' ) && isa( args_2, 'processing.signal_matrix' ) )
                errorStruct.message = 'args_1 and args_2 must be processing.signal_matrix!';
                errorStruct.identifier = 'times:NoSignalArrays';
                error( errorStruct );
            end

            % multiple args_1 / single args_2
            if ~isscalar( args_1 ) && isscalar( args_2 )
                args_2 = repmat( args_2, size( args_1 ) );
            end

            % single args_1 / multiple args_2
            if isscalar( args_1 ) && ~isscalar( args_2 )
                args_1 = repmat( args_1, size( args_2 ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( args_1, args_2 );

            %--------------------------------------------------------------
            % 2.) element-wise multiplications
            %--------------------------------------------------------------
            % iterate signal matrices
            for index_object = 1:numel( args_1 )

                % ensure identical axes
% TODO: allow margin of error!
                if any( abs( args_1( index_object ).axis.members - args_2( index_object ).axis.members ) >= 1e-10 * args_1( index_object ).axis.members )
%                 if ~isequal( args_1( index_object ).axis.members, args_2( index_object ).axis.members )
                    errorStruct.message = sprintf( 'args_1( %d ) and args_2( %d ) must have identical members in their axes!', index_object, index_object );
                    errorStruct.identifier = 'times:DiverseAxes';
                    error( errorStruct );
                end

                % perform element-wise multiplication
                args_1( index_object ).samples = args_1( index_object ).samples .* args_2( index_object ).samples;

                % update number of signals
                args_1( index_object ).N_signals = size( args_1( index_object ).samples, 2 );

            end % for index_object = 1:numel( args_1 )

        end % function args_1 = times( args_1, args_2 )

        %------------------------------------------------------------------
        % addition (overload plus method)
        %------------------------------------------------------------------
        function args_1 = plus( args_1, args_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix
            if ~( isa( args_1, 'processing.signal_matrix' ) && isa( args_2, 'processing.signal_matrix' ) )
                errorStruct.message = 'args_1 and args_2 must be processing.signal_matrix!';
                errorStruct.identifier = 'plus:NoSignalMatrices';
                error( errorStruct );
            end

            % ensure equal axes
            if ~isequal( args_1.axis, args_2.axis )
                errorStruct.message = 'args_1 and args_2 must have equal properties axis!';
                errorStruct.identifier = 'plus:DifferentAxes';
                error( errorStruct );
            end

            % multiple args_1 / single args_2
            if ~isscalar( args_1 ) && isscalar( args_2 )
                args_2 = repmat( args_2, size( args_1 ) );
            end

            % single args_1 / multiple args_2
            if isscalar( args_1 ) && ~isscalar( args_2 )
                args_1 = repmat( args_1, size( args_2 ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( args_1, args_2 );

            %--------------------------------------------------------------
            % 2.) perform addition
            %--------------------------------------------------------------
            % iterate signal matrices
            for index_object = 1:numel( args_1 )

                % add samples
                args_1( index_object ).samples = args_1( index_object ).samples + args_2( index_object ).samples;

                % update number of signals
                args_1( index_object ).N_signals = size( args_1( index_object ).samples, 2 );

            end % for index_object = 1:numel( args_1 )

        end % function args_1 = plus( args_1, args_2 )

        %------------------------------------------------------------------
        % subtraction (overload minus method)
        %------------------------------------------------------------------
        function args_1 = minus( args_1, args_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix
            if ~( isa( args_1, 'processing.signal_matrix' ) && isa( args_2, 'processing.signal_matrix' ) )
                errorStruct.message = 'args_1 and args_2 must be processing.signal_matrix!';
                errorStruct.identifier = 'minus:NoSignalMatrices';
                error( errorStruct );
            end

            % ensure equal axes
            if ~isequal( args_1.axis, args_2.axis )
                errorStruct.message = 'args_1 and args_2 must have equal properties axis!';
                errorStruct.identifier = 'minus:DifferentAxes';
                error( errorStruct );
            end

            % multiple args_1 / single args_2
            if ~isscalar( args_1 ) && isscalar( args_2 )
                args_2 = repmat( args_2, size( args_1 ) );
            end

            % single args_1 / multiple args_2
            if isscalar( args_1 ) && ~isscalar( args_2 )
                args_1 = repmat( args_1, size( args_2 ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( args_1, args_2 );

            %--------------------------------------------------------------
            % 2.) perform subtraction
            %--------------------------------------------------------------
            % iterate signal matrices
            for index_object = 1:numel( args_1 )

                % subtract samples
                args_1( index_object ).samples = args_1( index_object ).samples - args_2( index_object ).samples;

                % update number of signals
                args_1( index_object ).N_signals = size( args_1( index_object ).samples, 2 );

            end % for index_object = 1:numel( args_1 )

        end % function args_1 = minus( args_1, args_2 )

        %------------------------------------------------------------------
        % sum of array elements (overload sum method)
        %------------------------------------------------------------------
        function result = sum( signal_matrices, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix
            if ~isa( signal_matrices, 'processing.signal_matrix' )
                errorStruct.message = 'signal_matrices must be processing.signal_matrix!';
                errorStruct.identifier = 'sum:NoSignalMatrices';
                error( errorStruct );
            end

            % ensure equal axes
            if ~isequal( signal_matrices.axis )
                errorStruct.message = 'signal_matrices must have equal properties axis!';
                errorStruct.identifier = 'sum:DifferentAxes';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) perform summation
            %--------------------------------------------------------------
            result = signal_matrices( 1 );
            result.samples = sum( cat( 3, signal_matrices.samples ), 3 );
            result.N_signals = size( result.samples, 2 );

        end % function signal_matrices = sum( signal_matrices, varargin )

        %------------------------------------------------------------------
% TODO: cross-correlation (overload xcorr method)
        %------------------------------------------------------------------
%         function varargout = xcorr( varargin )
% 
%             %--------------------------------------------------------------
%             % 1.) check arguments
%             %--------------------------------------------------------------
%             
%         end

        %------------------------------------------------------------------
        % subsample
        %------------------------------------------------------------------
        function signal_matrices = subsample( signal_matrices, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
% TODO: implement early exit conditions!
            % return for single argument
            if nargin < 2
                return;
            end

            % ensure class processing.signal_matrix
            if ~isa( signal_matrices, 'processing.signal_matrix' )
                errorStruct.message = 'signal_matrices must be processing.signal_matrix!';
                errorStruct.identifier = 'subsample:NoSignalMatrices';
                error( errorStruct );
            end

            % ensure nonempty indices_axes
            if nargin >= 2 && ~isempty( varargin{ 1 } )
                indices_axes = varargin{ 1 };
            else
                indices_axes = cell( size( signal_matrices ) );
                for index_object = 1:numel( signal_matrices )
                    indices_axes{ index_object } = ( 1:abs( signal_matrices( index_object ).axis ) );
                end
            end

            % ensure cell array for indices_axes
            if ~iscell( indices_axes )
                indices_axes = { indices_axes };
            end

            % ensure nonempty indices_signals
            if nargin >= 3 && ~isempty( varargin{ 2 } )
                indices_signals = varargin{ 2 };
            else
                indices_signals = cell( size( signal_matrices ) );
                for index_object = 1:numel( signal_matrices )
                    indices_signals{ index_object } = ( 1:signal_matrices( index_object ).N_signals );
                end
            end

            % ensure cell array for indices_signals
            if ~iscell( indices_signals )
                indices_signals = { indices_signals };
            end

            % multiple signal_matrices / single indices_axes
            if ~isscalar( signal_matrices ) && isscalar( indices_axes )
                indices_axes = repmat( indices_axes, size( signal_matrices ) );
            end

            % single signal_matrices / multiple indices_axes
            if isscalar( signal_matrices ) && ~isscalar( indices_axes )
                signal_matrices = repmat( signal_matrices, size( indices_axes ) );
            end

            % multiple signal_matrices / single indices_signals
            if ~isscalar( signal_matrices ) && isscalar( indices_signals )
                indices_signals = repmat( indices_signals, size( signal_matrices ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_matrices, indices_axes, indices_signals );

            %--------------------------------------------------------------
            % 2.) perform subsampling
            %--------------------------------------------------------------
            % subsample axes
% TODO: problematic for different types of axes!
% TODO: maintain regular axis, if possible!
            axes_sub = subsample( reshape( [ signal_matrices.axis ], size( signal_matrices ) ), indices_axes );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % assign subsampled axes
                signal_matrices( index_object ).axis = axes_sub( index_object );

                % subsample samples
                signal_matrices( index_object ).samples = signal_matrices( index_object ).samples( indices_axes{ index_object }, indices_signals{ index_object } );

                % update N_signals
                signal_matrices( index_object ).N_signals = numel( indices_signals{ index_object } );

            end % for index_object = 1:numel( signal_matrices )

        end % function signal_matrices = subsample( signal_matrices, varargin )

        %------------------------------------------------------------------
        % cut out submatrix
        %------------------------------------------------------------------
        function signal_matrices = cut_out( signal_matrices, lbs, ubs, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix
            if ~isa( signal_matrices, 'processing.signal_matrix' )
                errorStruct.message = 'signal_matrices must be processing.signal_matrix!';
                errorStruct.identifier = 'cut_out:NoSignalMatrices';
                error( errorStruct );
            end

            % method cut_out in math.sequence_increasing ensures correct lbs and ubs

            % ensure nonempty indices_signals
            if nargin >= 4 && ~isempty( varargin{ 1 } )
                indices_signals = varargin{ 1 };
            else
                indices_signals = cell( size( signal_matrices ) );
                for index_object = 1:numel( signal_matrices )
                    indices_signals{ index_object } = ( 1:signal_matrices( index_object ).N_signals );
                end
            end

            % ensure cell array for indices_signals
            if ~iscell( indices_signals )
                indices_signals = { indices_signals };
            end

            % ensure nonempty settings_window
            if nargin >= 5 && ~isempty( varargin{ 2 } )
                settings_window = varargin{ 2 };
            else
                settings_window = auxiliary.setting_window( @tukeywin, 0 );
            end

            % multiple signal_matrices / single indices_signals
            if ~isscalar( signal_matrices ) && isscalar( indices_signals )
                indices_signals = repmat( indices_signals, size( signal_matrices ) );
            end

            % single signal_matrices / multiple indices_signals
            if isscalar( signal_matrices ) && ~isscalar( indices_signals )
                signal_matrices = repmat( signal_matrices, size( indices_signals ) );
            end

            % multiple signal_matrices / single settings_window
            if ~isscalar( signal_matrices ) && isscalar( settings_window )
                settings_window = repmat( settings_window, size( signal_matrices ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_matrices, indices_signals, settings_window );

            %--------------------------------------------------------------
            % 2.) perform cut out
            %--------------------------------------------------------------
            % cut out axes
            axes = reshape( [ signal_matrices.axis ], size( signal_matrices ) );
            [ axes_cut, indicators ] = cut_out( axes, lbs, ubs );

            % ensure cell array for indicators
            if ~iscell( indicators )
                indicators = { indicators };
            end

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % compute window
                samples_window = window( settings_window( index_object ).handle, sum( indicators{ index_object } ), settings_window( index_object ).parameters{ : } );

                % cut out samples
                signal_matrices( index_object ).axis = axes_cut( index_object );
                signal_matrices( index_object ).samples = samples_window .* signal_matrices( index_object ).samples( indicators{ index_object }, indices_signals{ index_object } );

                % update N_signals
                signal_matrices( index_object ).N_signals = numel( indices_signals{ index_object } );

            end % for index_object = 1:numel( signal_matrices )

        end % function signal_matrices = cut_out( signal_matrices, lbs, ubs, varargin )

        %------------------------------------------------------------------
        % return vector
        %------------------------------------------------------------------
        function y = return_vector( signal_matrices )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix
            if ~isa( signal_matrices, 'processing.signal_matrix' )
                errorStruct.message = 'signal_matrices must be processing.signal_matrix!';
                errorStruct.identifier = 'return_vector:NoSignalMatrices';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.physical_quantity
%             auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', signal_matrices.samples );

            %--------------------------------------------------------------
            % 2.) create and return vector
            %--------------------------------------------------------------
            % extract samples
            samples_cell = { signal_matrices.samples };

            % reshape matrices into column vectors
            for index_object = 1:numel( signal_matrices )
                samples_cell{ index_object } = samples_cell{ index_object }( : );
            end

            % concatenate column vectors
            y = cat( 1, samples_cell{ : } );

        end % function y = return_vector( signal_matrices )

        %------------------------------------------------------------------
        % energy
        %------------------------------------------------------------------
        function energies = energy( signal_matrices )

            % compute energy in samples
            energies = reshape( cellfun( @( x ) norm( x( : ) )^2, { signal_matrices.samples } ), size( signal_matrices ) );

        end % function energies = energy( signal_matrices )

        %------------------------------------------------------------------
        % data volume
        %------------------------------------------------------------------
        function volumes = data_volume( signal_matrices )

            % initialize data volumes
            volumes = physical_values.byte( zeros( size( signal_matrices ) ) );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                samples_act = signal_matrices( index_object ).samples;
                S = whos( 'samples_act' );
                volumes( index_object ) = physical_values.byte( S.bytes );

            end % for index_object = 1:numel( signal_matrices )

        end % function volumes = data_volume( signal_matrices )

        %------------------------------------------------------------------
        % 2-D line plot
        %------------------------------------------------------------------
        function hdl = show( signal_matrices )

            % initialize hdl with zeros
            hdl = zeros( size( signal_matrices ) );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % create new figure
                hdl( index_object ) = figure( index_object );

                % plot all samples
                if isreal( signal_matrices( index_object ).samples )
                    plot( double( signal_matrices( index_object ).axis.members ), double( signal_matrices( index_object ).samples.' ) );
                else
                    plot( double( signal_matrices( index_object ).axis.members ), double( abs( signal_matrices( index_object ).samples ).' ) );
                end

            end % for index_object = 1:numel( signal_matrices )

        end % function hdl = show( signal_matrices )

    end % methods

end % classdef signal_matrix