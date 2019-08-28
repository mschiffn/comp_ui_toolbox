% abstract superclass for all spatial discretization methods
%
% author: Martin F. Schiffner
% date: 2019-08-20
% modified: 2019-08-20
%
classdef (Abstract) options_spatial_method

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_spatial_method( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'options_spatial_method:NoRowVector';
                error( errorStruct );
            end

            % ensure positive integers
            mustBePositive( size );
            mustBeInteger( size );
            mustBeNonempty( size );

            %--------------------------------------------------------------
            % 2.) create options_spatial_method options
            %--------------------------------------------------------------
            % repeat options_spatial_method options
            objects = repmat( objects, size );

        end % function objects = options_spatial_method( size )

	end % methods

end % classdef (Abstract) options_spatial_method
