%
% abstract superclass for all spatial anti-aliasing filter options
%
% author: Martin F. Schiffner
% date: 2019-07-11
% modified: 2019-07-29
%
classdef (Abstract) options_anti_aliasing

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_anti_aliasing( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'options_anti_aliasing:NoRowVector';
                error( errorStruct );
            end

            % ensure positive integers
            mustBePositive( size );
            mustBeInteger( size );
            mustBeNonempty( size );

            %--------------------------------------------------------------
            % 2.) create spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % repeat spatial anti-aliasing filter options
            objects = repmat( objects, size );

        end % function objects = options_anti_aliasing( size )

	end % methods

end % classdef (Abstract) options_anti_aliasing
