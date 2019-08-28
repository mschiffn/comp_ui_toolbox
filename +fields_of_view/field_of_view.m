%
% superclass for all fields of view
%
% author: Martin F. Schiffner
% date: 2019-08-16
% modified: 2019-08-21
%
classdef field_of_view

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        shape ( 1, 1 ) geometry.shape { mustBeNonempty } = geometry.orthotope	% shape of the field of view

    end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = field_of_view( shapes )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return
            if nargin == 0
                return;
            end

            % ensure class geometry.shape
            if ~isa( shapes, 'geometry.shape' )
                errorStruct.message = 'shapes must be geometry.shape!';
                errorStruct.identifier = 'field_of_view:NoShapes';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create fields of view
            %--------------------------------------------------------------
            % repeat default field of view
            objects = repmat( objects, size( shapes ) );

            % iterate fields of view
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).shape = shapes( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = field_of_view( shapes )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function FOVs = discretize( FOVs, methods )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class fields_of_view.field_of_view
            if ~isa( FOVs, 'fields_of_view.field_of_view' )
                errorStruct.message = 'FOVs must be fields_of_view.field_of_view!';
                errorStruct.identifier = 'discretize:NoFieldsOfView';
                error( errorStruct );
            end

            % method discretize in shape ensures class discretizations.options_spatial_method for methods

            % multiple FOVs / single methods
            if ~isscalar( FOVs ) && isscalar( methods )
                methods = repmat( methods, size( FOVs ) );
            end

            % single FOVs / multiple methods
            if isscalar( FOVs ) && ~isscalar( methods )
                FOVs = repmat( FOVs, size( methods ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( FOVs, methods );

            %--------------------------------------------------------------
            % 2.) discretize fields of view
            %--------------------------------------------------------------
            % iterate fields of view
            for index_object = 1:numel( FOVs )

                % discretize shape
                FOVs( index_object ).shape = discretize( FOVs( index_object ).shape, methods( index_object ) );

            end % for index_object = 1:numel( FOVs )

        end % function FOVs = discretize( FOVs, methods )

    end % methods

end % classdef field_of_view
