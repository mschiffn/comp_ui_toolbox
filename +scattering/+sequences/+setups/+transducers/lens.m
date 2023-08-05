%
% abstract superclass for all acoustic lenses
%
% acoustic elevation lens:
% 1.) typically made of RTV silicone rubber (approx. 1025 m/s at 25°C)
% 2.) thickness >= 0.5 mm, thicker in the center
% 3.) used as protective layer
%
% author: Martin F. Schiffner
% date: 2019-08-15
% modified: 2019-10-17
%
classdef lens

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        axial_focus_axis ( 1, : ) physical_values.length { mustBeNonempty } = physical_values.meter( [ Inf, 16e-3 ] )	% axial distances of the lateral foci
        absorption_model ( 1, 1 ) scattering.sequences.setups.materials.absorption_models.absorption_model { mustBeNonempty } = scattering.sequences.setups.materials.absorption_models.none( physical_values.meter_per_second( 1540 ) )	% absorption model for the material

        % dependent properties
        thickness ( :, 1 ) { mustBeNonempty } = @scattering.sequences.setups.transducers.thickness.focus_axial	% thickness along each coordinate axis

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = lens( axial_focus_axis, absorption_models )
 
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for zero arguments
            if nargin == 0
                return;
            end

            % ensure cell array for axial_focus_axis
            if ~iscell( axial_focus_axis )
                axial_focus_axis = { axial_focus_axis };
            end

            % property validation function ensures scattering.sequences.setups.materials.absorption_models.absorption_model

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( axial_focus_axis, absorption_models );

            %--------------------------------------------------------------
            % 2.) create acoustic lenses
            %--------------------------------------------------------------
            % repeat default acoustic lens
            objects = repmat( objects, size( axial_focus_axis ) );

            % iterate acoustic lenses
            for index_object = 1:numel( objects )

                % ensure class physical_values.length
                if ~( isa( axial_focus_axis{ index_object }, 'physical_values.length' ) && isrow( axial_focus_axis{ index_object } ) )
                    errorStruct.message = sprintf( 'axial_focus_axis{ %d } must be physical_values.length!', index_object, index_object );
                    errorStruct.identifier = 'lens:NoLengths';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).axial_focus_axis = axial_focus_axis{ index_object };
                objects( index_object ).absorption_model = absorption_models( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = lens( axial_focus_axis, absorption_models )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function lenses = discretize( lenses, element_width_axis, positions_rel_norm )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.transducers.lens
            if ~isa( lenses, 'scattering.sequences.setups.transducers.lens' )
                errorStruct.message = 'lenses must be scattering.sequences.setups.transducers.lens!';
                errorStruct.identifier = 'discretize:NoLenses';
                error( errorStruct );
            end

            % ensure cell array for element_width_axis
            if ~iscell( element_width_axis )
                element_width_axis = { element_width_axis };
            end

            % ensure cell array for positions_rel_norm
            if ~iscell( positions_rel_norm )
                positions_rel_norm = { positions_rel_norm };
            end

            % multiple lenses / single positions_rel_norm
            if ~isscalar( lenses ) && isscalar( positions_rel_norm )
                positions_rel_norm = repmat( positions_rel_norm, size( lenses ) );
            end

            % single lenses / multiple positions_rel_norm
            if isscalar( lenses ) && ~isscalar( positions_rel_norm )
                lenses = repmat( lenses, size( positions_rel_norm ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( lenses, positions_rel_norm );

            %--------------------------------------------------------------
            % 2.) discretize acoustic lenses
            %--------------------------------------------------------------
            % iterate acoustic lenses
            for index_lens = 1:numel( lenses )

                % evaluate thickness
                lenses( index_lens ).thickness = lenses( index_lens ).thickness( lenses( index_lens ).axial_focus_axis, element_width_axis{ index_lens }, positions_rel_norm{ index_lens } );

            end % for index_lens = 1:numel( lenses )

        end % function lenses = discretize( lenses, positions_rel_norm )

	end % methods

end % classdef lens
