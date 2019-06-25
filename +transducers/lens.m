%
% superclass for all acoustic lens parameters
%
% author: Martin F. Schiffner
% date: 2019-06-03
% modified: 2019-06-05
%
classdef lens

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        thickness ( 1, 1 )
        absorption_model ( 1, 1 ) absorption_models.absorption_model = absorption_models.none( physical_values.meter_per_second( 1540 ) )	% absorption model for the acoustic lens

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = lens( axial_focus_axis, element_width_axis, absorption_models )

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

            % ensure cell array for element_width_axis
            if ~iscell( element_width_axis )
                element_width_axis = { element_width_axis };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( axial_focus_axis, element_width_axis, absorption_models );

            %--------------------------------------------------------------
            % 2.) create acoustic lens parameters
            %--------------------------------------------------------------
            % repeat default acoustic lens parameters
            objects = repmat( objects, size( axial_focus_axis ) );

            % iterate acoustic lens parameters
            for index_object = 1:numel( objects )

                % ensure class physical_values.length
                if ~( isa( axial_focus_axis{ index_object }, 'physical_values.length' ) && isrow( axial_focus_axis{ index_object } ) && isa( element_width_axis{ index_object }, 'physical_values.length' ) )
                    errorStruct.message = sprintf( 'axial_focus_axis{ %d } and element_width_axis{ %d } must be physical_values.length!', index_object, index_object );
                    errorStruct.identifier = 'lens:NoLengths';
                    error( errorStruct );
                end

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( axial_focus_axis{ index_object }, element_width_axis{ index_object } );

                % find elements that are infinite
                indicator_inf = isinf( axial_focus_axis{ index_object } );

                % normalized relative positions of the grid points
                axial_focus_axis_norm = 2 * axial_focus_axis{ index_object } ./ element_width_axis{ index_object };

                % maximum values
                max_vals = sqrt( 1 + axial_focus_axis_norm( ~indicator_inf ).^2 );

                % function handle for thickness profile
                thickness_hdl = @( pos_rel_norm ) sum( ( max_vals - sqrt( pos_rel_norm( :, ~indicator_inf ).^2 + axial_focus_axis_norm( ~indicator_inf ).^2 ) ) .* element_width_axis{ index_object }( ~indicator_inf ) / 2, 2 );

                % set independent properties
                objects( index_object ).thickness = thickness_hdl;
                objects( index_object ).absorption_model = absorption_models( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = lens( axial_focus_axis, element_width_axis, absorption_models )

	end % methods

end % classdef lens
