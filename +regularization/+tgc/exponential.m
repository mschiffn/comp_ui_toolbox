%
% superclass for all exponential time gain compensation (TGC) options
%
% author: Martin F. Schiffner
% date: 2019-12-19
% modified: 2020-04-03
%
classdef exponential < regularization.tgc.tgc

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        exponents ( :, 1 ) physical_values.frequency { mustBePositive, mustBeNonempty } = physical_values.hertz( 1 )
        decays_dB ( :, 1 ) double { mustBeNegative, mustBeNonempty } = -40

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = exponential( exponents, decays_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for exponents
            if ~iscell( exponents )
                exponents = { exponents };
            end

            % ensure cell array for exponents
            if ~iscell( decays_dB )
                decays_dB = { decays_dB };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( exponents, decays_dB );

            %--------------------------------------------------------------
            % 2.) create exponential TGC options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.tgc.tgc( size( exponents ) );

            % iterate exponential TGC options
            for index_object = 1:numel( objects )

                % property validation function ensures class physical_values.frequency for exponents{ index_object }
                % property validation function ensures nonempty negative doubles for decays_dB{ index_object }

                % multiple exponents{ index_object } / single decays_dB{ index_object }
                if ~isscalar( exponents{ index_object } ) && isscalar( decays_dB{ index_object } )
                    decays_dB{ index_object } = repmat( decays_dB{ index_object }, size( exponents{ index_object } ) );
                end

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( exponents{ index_object }, decays_dB{ index_object } );

                % set independent properties
                objects( index_object ).exponents = exponents{ index_object }( : );
                objects( index_object ).decays_dB = decays_dB{ index_object }( : );

            end % for index_object = 1:numel( objects )

        end % function objects = exponential( exponents, decays_dB )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( tgcs_exponential )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.tgc.exponential
            if ~isa( tgcs_exponential, 'regularization.tgc.exponential' )
                errorStruct.message = 'tgcs_exponential must be regularization.tgc.exponential!';
                errorStruct.identifier = 'string:NoTGCExponential';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % initializse string array for strs_out
            strs_out = repmat( "", size( tgcs_exponential ) );

            % iterate exponential TGC options
            for index_object = 1:numel( tgcs_exponential )

                strs_out( index_object ) = sprintf( "%s", 'exponential' );

            end % for index_object = 1:numel( tgcs_exponential )

        end % function strs_out = string( tgcs_exponential )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % create linear transform (scalar)
        %------------------------------------------------------------------
        function [ LT, LTs_measurement ] = get_LT_scalar( tgc, operator_born )
% TODO: no linear_transforms.concatenations.diagonal for single mixed RF voltage signal? single convolution?
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class regularization.tgc.tgc (scalar)
            % calling function ensures class scattering.operator_born (scalar)

            %--------------------------------------------------------------
            % 2.) create linear transform (scalar)
            %--------------------------------------------------------------
            % numbers of observations for all sequential pulse-echo measurements
            N_observations_mix = { operator_born.sequence.settings( operator_born.indices_measurement_sel ).N_observations };

            % number of mixed voltage signals for each sequential pulse-echo measurement
            N_mixes_measurement = cellfun( @numel, N_observations_mix );

            % specify cell array for LTs_measurement
            LTs_measurement = cell( numel( operator_born.indices_measurement_sel ), 1 );

            % indices for each mix
            indices = mat2cell( 1:sum( N_mixes_measurement ), 1, N_mixes_measurement );

            %------------------------------------------------------
            % a) extract frequency axes, time intervals, and numbers observations
            %------------------------------------------------------
            % specify cell arrays
            axes_f_mix = cell( numel( operator_born.indices_measurement_sel ), 1 );
            intervals_t = cell( numel( operator_born.indices_measurement_sel ), 1 );

            % iterate selected sequential pulse-echo measurements
            for index_measurement_sel = 1:numel( operator_born.indices_measurement_sel )

                % index of sequential pulse-echo measurement
                index_measurement = operator_born.indices_measurement_sel( index_measurement_sel );

                % subsample global unique frequencies to get unique frequencies of pulse-echo measurement
                axis_f_measurement_unique = subsample( operator_born.sequence.axis_f_unique, operator_born.sequence.indices_f_to_unique( index_measurement ) );

                % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
                indices_f_mix_to_measurement = operator_born.sequence.settings( index_measurement ).indices_f_to_unique;

                % subsample unique frequencies of pulse-echo measurement to get frequencies of mixed voltage signals
                axes_f_mix{ index_measurement_sel } = subsample( axis_f_measurement_unique, indices_f_mix_to_measurement );

                intervals_t{ index_measurement_sel } = [ operator_born.sequence.settings( index_measurement ).rx.interval_t ].';

            end % for index_measurement_sel = 1:numel( operator_born.indices_measurement_sel )

            % concatenate vertically
            axes_f_mix = cat( 1, axes_f_mix{ : } );
            intervals_t = cat( 1, intervals_t{ : } );

            %--------------------------------------------------------------
            % A) exponential TGC curves
            %--------------------------------------------------------------
            TGC_curves = regularization.tgc.curves.exponential( intervals_t, tgc.exponents );

            %--------------------------------------------------------------
            % c) create discrete convolutions by discretizing TGC curves
            %--------------------------------------------------------------
            % time intervals for discretization
            Ts_ref = reshape( 1 ./ [ axes_f_mix.delta ], size( axes_f_mix ) );

            % compute Fourier coefficients
            signal_matrices = fourier_coefficients( TGC_curves, Ts_ref, tgc.decays_dB );

            % compute kernels for discrete convolutions
            kernels = cell( size( signal_matrices ) );
            for index_mix = 1:numel( signal_matrices )

                kernels{ index_mix } = [ conj( signal_matrices( index_mix ).samples( end:-1:2 ) ); signal_matrices( index_mix ).samples ];

            end % for index_mix = 1:numel( signal_matrices )

            % create discrete convolution for each mix
            LTs_conv = num2cell( linear_transforms.convolutions.fft( kernels, cat( 1, N_observations_mix{ : } ) ) );

            %--------------------------------------------------------------
            % d) concatenate discrete convolutions diagonally
            %--------------------------------------------------------------
            % create TGC operator for each selected sequential pulse-echo measurement
            for index_measurement_sel = 1:numel( operator_born.indices_measurement_sel )
                LTs_measurement{ index_measurement_sel } = linear_transforms.concatenations.diagonal( LTs_conv{ indices{ index_measurement_sel } } );
            end

            % create TGC operator for all selected sequential pulse-echo measurement
            LT = linear_transforms.concatenations.diagonal( LTs_conv{ : } );

            % concatenate vertically
            LTs_measurement = cat( 1, LTs_measurement{ : } );

        end % function [ LT, LTs_measurement ] = get_LT_scalar( tgc, operator_born )

	end % methods (Access = protected, Hidden)

end % classdef exponential < regularization.tgc.tgc
