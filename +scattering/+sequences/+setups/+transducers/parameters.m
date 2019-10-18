%
% superclass for all transducer array parameters
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-06-03
%
classdef parameters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_elements_axis ( 1, : ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 128, 1 ]	% numbers of elements along each coordinate axis (1)
        apodization ( 1, : )                                % apodization function (r = normalized relative positions of the grid points)
        axial_focus_axis ( 1, : ) physical_values.length	% axial distances of the foci along each coordinate axis
        str_model = 'Default Array'             % model name
        str_vendor = 'Default Corporation'      % vendor name

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters( N_elements_axis, apodization, axial_focus_axis, str_model, str_vendor )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for zero arguments
            if nargin == 0
                return;
            end

            % ensure cell array for N_elements_axis
            if ~iscell( N_elements_axis )
                N_elements_axis = { N_elements_axis };
            end

            % ensure cell array for apodization
            if ~iscell( apodization )
                apodization = { apodization };
            end

            % ensure cell array for axial_focus_axis
            if ~iscell( axial_focus_axis )
                axial_focus_axis = { axial_focus_axis };
            end

            % ensure cell array for str_model
            if ~iscell( str_model )
                str_model = { str_model };
            end

            % ensure cell array for str_name
            if ~iscell( str_vendor )
                str_vendor = { str_vendor };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( N_elements_axis, apodization, axial_focus_axis, str_model, str_vendor );

            %--------------------------------------------------------------
            % 2.) create transducer array parameters
            %--------------------------------------------------------------
            objects = repmat( objects, size( N_elements_axis ) );

            % check and set independent properties
            for index_object = 1:numel( objects )

                % ensure scalar or row vector for apodization
                if ~( isscalar( apodization{ index_object } ) || auxiliary.isEqualSize( apodization{ index_object }, N_elements_axis{ index_object } ) )
                    errorStruct.message = sprintf( 'The size of apodization{ %d } must be scalar or match that of N_elements_axis{ %d }', index_object, index_object );
                    errorStruct.identifier = 'parameters:SizeMismatch';
                    error( errorStruct );
                end

% TODO: ensure function handles

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( axial_focus_axis{ index_object }, N_elements_axis{ index_object } );

                % set independent properties
                objects( index_object ).N_elements_axis = N_elements_axis{ index_object };
                objects( index_object ).apodization = apodization{ index_object };
                objects( index_object ).axial_focus_axis = axial_focus_axis{ index_object };
                objects( index_object ).str_model = str_model{ index_object };
                objects( index_object ).str_vendor = str_vendor{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = parameters( N_elements_axis, apodization, axial_focus_axis, str_model, str_vendor )

        %------------------------------------------------------------------
        % project
        %------------------------------------------------------------------
        function parameters = project( parameters, N_dimensions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure positive integers
            mustBeInteger( N_dimensions );
            mustBePositive( N_dimensions );

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( parameters, N_dimensions );

            %--------------------------------------------------------------
            % 2.) project parameters onto lower or equal dimension
            %--------------------------------------------------------------
            for index_object = 1:numel( parameters )

                % consider maximum number of dimensions in parameters
                if N_dimensions( index_object ) > numel( parameters( index_object ).N_elements_axis )
                    errorStruct.message     = sprintf( 'N_dimensions( %d ) exceeds the number of specified dimensions in parameters( %d )!', index_object, index_object );
                    errorStruct.identifier	= 'project:LargeNumberDimensions';
                    error( errorStruct );
                end

                % extract relevant numbers of elements
                parameters( index_object ).N_elements_axis = parameters( index_object ).N_elements_axis( 1:N_dimensions( index_object ) );

                % extract relevant components of apodization function
                if ~isscalar( parameters( index_object ).apodization )
                    parameters( index_object ).apodization = parameters( index_object ).apodization{ 1:N_dimensions( index_object ) };
                end

                % extract relevant components of axial_focus_axis
                parameters( index_object ).axial_focus_axis = parameters( index_object ).axial_focus_axis( 1:N_dimensions( index_object ) );

            end % for index_object = 1:numel( parameters )

        end % function parameters = project( parameters, N_dimensions )

        %------------------------------------------------------------------
        % number of elements
        %------------------------------------------------------------------
        function results = N_elements( parameters )

            % compute numbers of elements
            results = reshape( cellfun( @prod, { parameters.N_elements_axis } ), size( parameters ) );

        end % function results = N_elements( parameters )

	end % methods

end % classdef parameters
