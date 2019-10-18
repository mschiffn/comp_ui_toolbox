%
% none (lossless) absorption model
%
% author: Martin F. Schiffner
% date: 2016-08-25
% modified: 2019-10-17
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

        %------------------------------------------------------------------
        % compute complex-valued wavenumbers (implement abstract compute_wavenumbers method)
        %------------------------------------------------------------------
        function axes_k_tilde = compute_wavenumbers( nones, axes_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.sequence_increasing
% TODO: ensure physical_values.frequency
            if ~isa( axes_f, 'math.sequence_increasing' )
                axes_f = math.sequence_increasing( axes_f );
            end

            % multiple nones / single axes_f
            if ~isscalar( nones ) && isscalar( axes_f )
                axes_f = repmat( axes_f, size( nones ) );
            end

            % single nones / multiple axes_f
            if isscalar( nones ) && ~isscalar( axes_f )
                nones = repmat( nones, size( axes_f ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( nones, axes_f );

            %--------------------------------------------------------------
            % 2.) compute complex-valued wavenumbers
            %--------------------------------------------------------------
            % specify cell array for axes_k_tilde
            axes_k_tilde = cell( size( nones ) );

            % iterate time causal absorption models
            for index_object = 1:numel( nones )

                % compose complex-valued wavenumbers
                axes_k_tilde{ index_object } = 2 * pi * axes_f( index_object ).members / nones( index_object ).c_0;

            end % for index_object = 1:numel( nones )

            %--------------------------------------------------------------
            % 3.) create increasing sequences
            %--------------------------------------------------------------
            axes_k_tilde = math.sequence_increasing( axes_k_tilde );

        end % function axes_k_tilde = compute_wavenumbers( nones, axes_f )

    end % methods

end % classdef none < scattering.sequences.setups.materials.absorption_models.absorption_model
