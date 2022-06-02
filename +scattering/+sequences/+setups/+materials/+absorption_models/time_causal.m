%
% time causal absorption model
% (Szabo 2004, Waters 2005, Kelly2006)
%
% author: Martin F. Schiffner
% date: 2016-08-25
% modified: 2021-05-21
%
classdef time_causal < scattering.sequences.setups.materials.absorption_models.absorption_model

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        absorption_constant             % dB / cm
        absorption                      % dB / (MHz^exponent * cm)
        exponent ( 1, 1 ) double        % exponent in power law (1)
        c_ref ( 1, 1 ) physical_values.velocity     % reference phase velocity (m/s)
        f_ref ( 1, 1 ) physical_values.frequency	% temporal reference frequency for reference phase velocity c_ref (Hz)
        flag_dispersion = 1;            % include frequency-dependent dispersion (causal model) if nonzero

        % dependent properties
        alpha_0                         % Np / m
        alpha_1                         % Np / (Hz^exponent * m)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = time_causal( constants, slopes, exponents, c_ref, f_ref, flag_dispersion )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure at least five and at most six arguments
            narginchk( 5, 6 );

            % TODO: constants in dB / cm
            % TODO: slopes in dB / (MHz^exponents * cm)

            % ensure existence of nonempty flag_dispersion
            if nargin < 6 || isempty( flag_dispersion )
                flag_dispersion = true( size( constants ) );
            end

            % ensure equal number of dimensions and sizes
            [ constants, slopes, exponents, c_ref, f_ref, flag_dispersion ] = auxiliary.ensureEqualSize( constants, slopes, exponents, c_ref, f_ref, flag_dispersion );

            %--------------------------------------------------------------
            % 2.) create time causal absorption models
            %--------------------------------------------------------------
            % create name string
            strs_name = repmat( { 'power_law' }, size( constants ) );

            % constructor of superclass
            objects@scattering.sequences.setups.materials.absorption_models.absorption_model( strs_name );

            % iterate time causal absorption models
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).absorption_constant = constants( index_object );
                objects( index_object ).absorption = slopes( index_object );
                objects( index_object ).exponent = exponents( index_object );
                objects( index_object ).c_ref = c_ref( index_object );
                objects( index_object ).f_ref = f_ref( index_object );
                objects( index_object ).flag_dispersion = flag_dispersion( index_object );

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                objects( index_object ).alpha_0 = objects( index_object ).absorption_constant * log( 10 ) / ( 20 * physical_values.meter( 0.01 ) );
                objects( index_object ).alpha_1 = objects( index_object ).absorption * log( 10 ) / ( 20 * physical_values.meter( 0.01 ) * physical_values.hertz( 1e6 ).^objects( index_object ).exponent );

            end % for index_object = 1:numel( objects )

        end % function objects = time_causal( constants, slopes, exponents, c_ref, f_ref, flag_dispersion )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute complex-valued wavenumbers (scalar)
        %------------------------------------------------------------------
        function samples_k_tilde = compute_wavenumbers_scalar( time_causal, axis_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.setups.materials.absorption_models.absorption_model for time_causal
            % calling function ensures class math.sequence_increasing for axis_f

            % ensure class scattering.sequences.setups.materials.absorption_models.time_causal
            if ~isa( time_causal, 'scattering.sequences.setups.materials.absorption_models.time_causal' )
                errorStruct.message = 'time_causal must be scattering.sequences.setups.materials.absorption_models.time_causal!';
                errorStruct.identifier = 'compute_wavenumbers_scalar:NoTimeCausalModel';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute complex-valued wavenumbers (scalar)
            %--------------------------------------------------------------
            % compute real-valued wavenumbers using reference phase velocity
            samples_k_ref = 2 * pi * axis_f.members / time_causal.c_ref;

            % compute imaginary part (even function of temporal frequency)
            samples_k_tilde_imag = - time_causal.alpha_0 - time_causal.alpha_1 * abs( axis_f.members ).^time_causal.exponent;

            % compute real part (odd function of temporal frequency)
            if time_causal.flag_dispersion

                %----------------------------------------------------------
                % a) include frequency-dependent dispersion (causal model)
                %----------------------------------------------------------
                if mod( time_causal.exponent, 2 ) == 0 || floor( time_causal.exponent ) ~= time_causal.exponent

                    % exponent is an even integer or noninteger
                    samples_k_tilde_real = samples_k_ref + time_causal.alpha_1 * tan( time_causal.exponent * pi / 2 ) * axis_f.members .* ( abs( axis_f.members ).^( time_causal.exponent - 1 ) - abs( time_causal.f_ref )^( time_causal.exponent - 1 ) );
                else

                    % exponent is an odd integer
                    samples_k_tilde_real = samples_k_ref - 2 * time_causal.alpha_1 * axis_f.members.^time_causal.exponent .* log( abs( axis_f.members / time_causal.f_ref ) ) / pi;
                end

            else

                %----------------------------------------------------------
                % b) ignore frequency-dependent dispersion (noncausal model)
                %----------------------------------------------------------
                samples_k_tilde_real = samples_k_ref;

            end % if time_causal.flag_dispersion

            % check for zero frequency and ensure odd function of temporal frequency
            indicator = double( abs( axis_f.members ) ) < eps;
            samples_k_tilde_real( indicator ) = 0;

            % compose complex-valued wavenumbers
            samples_k_tilde = samples_k_tilde_real + 1j * samples_k_tilde_imag;

        end % function samples_k_tilde = compute_wavenumbers_scalar( time_causal, axis_f )

	end % methods (Access = protected, Hidden)

end % classdef time_causal < scattering.sequences.setups.materials.absorption_models.absorption_model
