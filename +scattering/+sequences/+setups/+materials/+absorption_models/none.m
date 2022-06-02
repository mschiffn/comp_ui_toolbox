%
% none (lossless) absorption model
%
% author: Martin F. Schiffner
% date: 2016-08-25
% modified: 2021-05-21
%
classdef none < scattering.sequences.setups.materials.absorption_models.absorption_model

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        c_0 ( 1, 1 ) physical_values.velocity	% constant phase and group velocity

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = none( c_0 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure class physical_values.velocity

            %--------------------------------------------------------------
            % 2.) create lossless absorption model
            %--------------------------------------------------------------
            % create name strings
            strs_name = repmat( { 'none' }, size( c_0 ) );

            % constructor of superclass
            objects@scattering.sequences.setups.materials.absorption_models.absorption_model( strs_name );

            % iterate lossless absorption models
            for index_object = 1:numel( objects )

                % internal properties
                objects( index_object ).c_0 = c_0( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = none( c_0 )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute complex-valued wavenumbers (scalar)
        %------------------------------------------------------------------
        function samples_k_tilde = compute_wavenumbers_scalar( none, axis_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.setups.materials.absorption_models.absorption_model for none
            % calling function ensures class math.sequence_increasing for axis_f

            % ensure class scattering.sequences.setups.materials.absorption_models.none
            if ~isa( none, 'scattering.sequences.setups.materials.absorption_models.none' )
                errorStruct.message = 'none must be scattering.sequences.setups.materials.absorption_models.none!';
                errorStruct.identifier = 'compute_wavenumbers_scalar:NoNoneModel';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute complex-valued wavenumbers (scalar)
            %--------------------------------------------------------------
            % compose complex-valued wavenumbers
            samples_k_tilde = 2 * pi * axis_f.members / none.c_0;

        end % function samples_k_tilde = compute_wavenumbers_scalar( none, axis_f )

	end % methods (Access = protected, Hidden)

end % classdef none < scattering.sequences.setups.materials.absorption_models.absorption_model
