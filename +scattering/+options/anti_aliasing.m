%
% abstract superclass for all spatial anti-aliasing filter options
%
% author: Martin F. Schiffner
% date: 2019-07-11
% modified: 2019-08-03
%
classdef (Abstract) anti_aliasing

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = anti_aliasing( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'anti_aliasing:NoRowVector';
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

        end % function objects = anti_aliasing( size )

	end % methods

end % classdef (Abstract) anti_aliasing
