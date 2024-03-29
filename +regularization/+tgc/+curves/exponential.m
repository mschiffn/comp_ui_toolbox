%
% superclass for all exponential time gain compensation curves
%
% author: Martin F. Schiffner
% date: 2019-12-07
% modified: 2020-06-30
%
classdef exponential < regularization.tgc.curves.curve

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        exponent ( 1, 1 ) physical_values.frequency { mustBePositive, mustBeNonempty } = physical_values.hertz( 1 )

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = exponential( intervals_t, exponents )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class math.interval for intervals_t
            % superclass ensures equal subclasses of physical_values.time for bounds in intervals_t

            % property validation functions ensure nonempty positive frequencies for exponents

            % ensure two arguments
            narginchk( 2, 2 );

            % ensure equal number of dimensions and sizes
            [ intervals_t, exponents ] = auxiliary.ensureEqualSize( intervals_t, exponents );

            %--------------------------------------------------------------
            % 2.) create exponential time gain compensation curves
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.tgc.curves.curve( intervals_t );

            % iterate exponential time gain compensation curves
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).exponent = exponents( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = exponential( intervals_t, exponents )

        %------------------------------------------------------------------
        % sample exponential time gain compensation curves
        %------------------------------------------------------------------
        function samples = sample_curve( curves, axes )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure class regularization.tgc.curves.exponential
            if ~isa( curves, 'regularization.tgc.curves.exponential' )
                errorStruct.message = 'tgcs must be regularization.tgc.curves.exponential!';
                errorStruct.identifier = 'compute_samples:NoExponentialTGC';
                error( errorStruct );
            end

            % ensure class math.sequence_increasing
            if ~isa( axes, 'math.sequence_increasing' )
                errorStruct.message = 'axes must be math.sequence_increasing!';
                errorStruct.identifier = 'compute_samples:NoSequenceIncreasing';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            [ curves, axes ] = auxiliary.ensureEqualSize( curves, axes );

            %--------------------------------------------------------------
            % 2.) sample exponential time gain compensation curves
            %--------------------------------------------------------------
            % specify cell array for samples
            samples = cell( size( curves ) );

            % iterate exponential time gain compensation curves
            for index_object = 1:numel( curves )

                if isa( axes( index_object ).members, 'physical_values.time' )

                    %------------------------------------------------------
                    % a) time-domain samples
                    %------------------------------------------------------
                    % set independent properties
                    samples{ index_object } = exp( curves( index_object ).exponent * ( axes( index_object ).members - curves( index_object ).interval_t.lb ) );

                elseif isa( axes( index_object ).members, 'physical_values.frequency' )

                    %------------------------------------------------------
                    % b) frequency-domain samples
                    %------------------------------------------------------
                    temp_1 = curves( index_object ).exponent - 2j * pi * axes( index_object ).members;
                    arg_1 = temp_1 * curves( index_object ).T / 2;
                    phase = exp( - 2j * pi * axes( index_object ).members * curves( index_object ).interval_t.lb );
                    samples{ index_object } = 2 * exp( arg_1 ) .* sinh( arg_1 ) .* phase ./ ( temp_1 * curves( index_object ).T );

                end % if isa( axes( index_object ).members, 'physical_values.time' )

            end % for index_object = 1:numel( curves )

            % avoid cell array for single curves
            if isscalar( curves )
                samples = samples{ 1 };
            end

        end % function samples = sample_curve( curves, axes )

        %------------------------------------------------------------------
        % Fourier coefficients
        %------------------------------------------------------------------
        function signal_matrices = fourier_coefficients( curves, T_ref, decays_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.tgc.curves.exponential
            if ~isa( curves, 'regularization.tgc.curves.exponential' )
                errorStruct.message = 'curves must be regularization.tgc.curves.exponential!';
                errorStruct.identifier = 'fourier_coefficients:NoExponentialCurves';
                error( errorStruct );
            end

            % ensure nonempty T_ref
            if nargin < 2 || isempty( T_ref )
                T_ref = reshape( [ curves.T ], size( curves ) );
            end

            % ensure class physical_values.time
            

            % ensure valid T_ref
            indicator = T_ref < reshape( [ curves.T ], size( curves ) );
            if any( indicator( : ) )
                errorStruct.message = 'T_ref must be greater than or equal to T!';
                errorStruct.identifier = 'fourier_coefficients:InvalidReferenceT';
                error( errorStruct );
            end

            % ensure nonempty decays_dB
            if nargin < 3 || isempty( decays_dB )
                decays_dB = -40;
            end

            % ensure negative decays_dB
            mustBeNegative( decays_dB );

            % ensure equal number of dimensions and sizes
            [ curves, T_ref, decays_dB ] = auxiliary.ensureEqualSize( curves, T_ref, decays_dB );

            %--------------------------------------------------------------
            % 2.) compute Fourier coefficients
            %--------------------------------------------------------------
            % compute upper frequency bounds
            if abs( T_ref - reshape( [ curves.T ], size( curves ) ) ) <= eps( 0 ) * [ curves.T ]
                ubs_f = reshape( [ curves.exponent ], size( curves ) ) .* sqrt( 10.^( - decays_dB / 10 ) - 1 ) / ( 2 * pi );
            else
% TODO: exponent close to zero?
                temp_1 = reshape( exp( - [ curves.exponent ] .* [ curves.T ] ), size( curves ) );
                ubs_f = reshape( [ curves.exponent ], size( curves ) ) .* sqrt( ( ( 1 + temp_1 ) .* 10.^( - decays_dB / 20 ) ./ ( 1 - temp_1 ) ).^2 - 1 ) / ( 2 * pi );
            end

            % create frequency axes
            axes_f = math.sequence_increasing_regular_quantized( zeros( size( curves ) ), ceil( ubs_f .* T_ref ), 1 ./ T_ref );

            % specify cell array for samples
            samples = cell( size( curves ) );

            % iterate exponential time gain compensation curves
            for index_object = 1:numel( curves )

                denominator = curves( index_object ).exponent - 2j * pi * axes_f( index_object ).members;
                numerator = exp( denominator * curves( index_object ).T ) - 1;
                phase_shift = exp( - 2j * pi * axes_f( index_object ).members * curves( index_object ).interval_t.lb );

                samples{ index_object } = phase_shift .* numerator ./ ( denominator * T_ref( index_object ) );

            end % for index_object = 1:numel( curves )

            % create signal matrices
            signal_matrices = processing.signal_matrix( axes_f, samples );

        end % function signal_matrices = fourier_coefficients( curves, T_ref, decays_dB )

        %------------------------------------------------------------------
        % get axes
        %------------------------------------------------------------------
        function axes_f = get_axes( curves, decays_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.tgc.curves.exponential
            if ~isa( curves, 'regularization.tgc.curves.exponential' )
                errorStruct.message = 'curves must be regularization.tgc.curves.exponential!';
                errorStruct.identifier = 'compute_samples:NoExponentialTGC';
                error( errorStruct );
            end

            % ensure nonempty negative decays_dB
            mustBeNonempty( decays_dB );
            mustBeNegative( decays_dB );

            % multiple curves / single decays_dB
            if ~isscalar( curves ) && isscalar( decays_dB )
                decays_dB = repmat( decays_dB, size( curves ) );
            end

            % single curves / multiple decays_dB
            if isscalar( curves ) && ~isscalar( decays_dB )
                curves = repmat( curves, size( decays_dB ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( curves, decays_dB );

            %--------------------------------------------------------------
            % 2.) get axes
            %--------------------------------------------------------------
            % compute upper frequency bounds
            ubs_f = [ curves.exponent ] .* sqrt( ( ( 1 + exp( - [ curves.exponent ] .* [ curves.T ] ) ) .* 10.^( - decays_dB / 20 ) ./ ( 1 - exp( - [ curves.exponent ] .* [ curves.T ] ) ) ).^2 - 1 ) / ( 2 * pi );

            % create axes
            axes_f = math.sequence_increasing_regular_quantized( zeros( size( curves ) ), ceil( ubs_f .* reshape( [ curves.T ], size( curves ) ) ), 1 ./ reshape( [ curves.T ], size( curves ) ) );

        end % function axes_f = get_axes( curves, decays_dB )

	end % methods

end % classdef exponential < regularization.tgc.curves.curve
