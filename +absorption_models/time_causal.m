%
% time causal absorption model
% (Szabo 2004, Waters 2005, Kelly2006)
%
% author: Martin F. Schiffner
% date: 2016-08-25
% modified: 2019-04-22
%
classdef time_causal < absorption_models.absorption_model

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
        function objects = time_causal( constants, slopes, exponents, c_ref, f_ref, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure sufficient number of arguments
            if nargin < 5
                errorStruct.message     = 'At least 5 arguments are required!';
                errorStruct.identifier	= 'time_causal:Arguments';
                error( errorStruct );
            end

            % TODO: constants in dB / cm
            % TODO: slopes in dB / (MHz^exponents * cm)
            %
            if nargin >= 6
                flag_dispersion = varargin{ 1 };
            end

            % multiple constants / single flag_dispersion
            if ~isscalar( constants ) && isscalar( flag_dispersion )
                flag_dispersion = repmat( flag_dispersion, size( constants ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( constants, slopes, exponents, c_ref, f_ref, flag_dispersion );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            % create name string
            str_name = sprintf( 'power_law_%.2f_%.2f_%.2f', constants, slopes, exponents );

            objects@absorption_models.absorption_model( str_name );

            %--------------------------------------------------------------
            % 3.) set properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).absorption_constant = constants( index_object );
                objects( index_object ).absorption = slopes( index_object );
                objects( index_object ).exponent = exponents( index_object );
                objects( index_object ).c_ref = c_ref( index_object );
                objects( index_object ).f_ref = f_ref( index_object );
                objects( index_object ).flag_dispersion = flag_dispersion( index_object );

                % set dependent properties
                objects( index_object ).alpha_0 = objects( index_object ).absorption_constant * log( 10 ) / ( 20 * physical_values.meter( 0.01 ) );
                objects( index_object ).alpha_1 = objects( index_object ).absorption * log( 10 ) / ( 20 * physical_values.meter( 0.01 ) * physical_values.hertz( 1e6 ).^objects( index_object ).exponent );

            end % for index_object = 1:numel( objects )

        end % function objects = time_causal( constants, slopes, exponents, c_ref, f_ref, varargin )

        %------------------------------------------------------------------
        % compute complex-valued wavenumbers
        %------------------------------------------------------------------
        function axes_k_tilde = compute_wavenumbers( time_causal, axes_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.sequence_increasing
            if ~isa( axes_f, 'math.sequence_increasing' )
                axes_f = math.sequence_increasing( axes_f );
            end

            % multiple time_causal / single axes_f
            if ~isscalar( time_causal ) && isscalar( axes_f )
                axes_f = repmat( axes_f, size( time_causal ) );
            end

            % single time_causal / multiple axes_f
            if isscalar( time_causal ) && ~isscalar( axes_f )
                time_causal = repmat( time_causal, size( axes_f ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( time_causal, axes_f );

            %--------------------------------------------------------------
            % 2.) compute complex-valued wavenumbers
            %--------------------------------------------------------------
            % specify cell array
            axes_k_tilde = cell( size( time_causal ) );

            % iterate time causal absorption models
            for index_object = 1:numel( time_causal )

                %----------------------------------------------------------
                % a) compute real-valued wavenumbers using reference phase velocity
                %----------------------------------------------------------
                axis_k_ref = 2 * pi * axes_f( index_object ).members / time_causal( index_object ).c_ref;

                %----------------------------------------------------------
                % b) compute imaginary part (even function of temporal frequency)
                %----------------------------------------------------------
                axis_k_tilde_imag = - time_causal( index_object ).alpha_0 - time_causal( index_object ).alpha_1 * abs( axes_f( index_object ).members ).^time_causal( index_object ).exponent;

                %----------------------------------------------------------
                % c) compute real part (odd function of temporal frequency)
                %----------------------------------------------------------
                if time_causal( index_object ).flag_dispersion

                    % include frequency-dependent dispersion (causal model)
                    if mod( time_causal( index_object ).exponent, 2 ) == 0 || floor( time_causal( index_object ).exponent ) ~= time_causal( index_object ).exponent

                        % exponent is an even integer or noninteger
                        axis_k_tilde_real = axis_k_ref + time_causal( index_object ).alpha_1 * tan( time_causal( index_object ).exponent * pi / 2 ) * axes_f( index_object ).members .* ( abs( axes_f( index_object ).members ).^( time_causal( index_object ).exponent - 1 ) - abs( time_causal( index_object ).f_ref )^( time_causal( index_object ).exponent - 1 ) );
                    else

                        % exponent is an odd integer
                        axis_k_tilde_real = axis_k_ref - 2 * time_causal( index_object ).alpha_1 * axes_f( index_object ).members.^time_causal( index_object ).exponent .* log( abs( axes_f( index_object ).members / time_causal( index_object ).f_ref ) ) / pi;
                    end
                else

                    % ignore frequency-dependent dispersion (noncausal model)
                    axis_k_tilde_real = axis_k_ref;
                end

                % check for zero frequency and ensure odd function of temporal frequency
                indicator = double( abs( axes_f( index_object ).members ) ) < eps;
                axis_k_tilde_real( indicator ) = 0;

                % compose complex-valued wavenumbers
                axes_k_tilde{ index_object } = axis_k_tilde_real + 1j * axis_k_tilde_imag;

            end % for index_object = 1:numel( time_causal )

            %--------------------------------------------------------------
            % 3.) create increasing sequences
            %--------------------------------------------------------------
            axes_k_tilde = math.sequence_increasing( axes_k_tilde );

        end % function axes_k_tilde = compute_wavenumbers( time_causal, axes_f )

    end % methods

end % classdef time_causal < absorption_models.absorption_model
